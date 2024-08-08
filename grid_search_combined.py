# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import os
import pickle
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from apex.optimizers import FusedAdam
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp

import numpy as np

import dllogger

from modeling import AltTemporalFusionTransformer
from configuration import CONFIGS, get_attention_names, get_attention_hparam_grid, make_attn_module_class
from data_utils import load_dataset
from log_helper import setup_logger
from criterions import QuantileLoss
from inference import predict
from utils import PerformanceMeter, print_once
import gpu_affinity
from ema import ModelEma

from itertools import product

def main(args):
    ### INIT DISTRIBUTED
    args.distributed_world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.distributed_world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        print_once(f'Distributed training with {args.distributed_world_size} GPUs')
        args.distributed_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
        torch.cuda.synchronize()

    # Enable CuDNN autotuner
    nproc_per_node = torch.cuda.device_count()
    if args.affinity != 'disabled':
        affinity = gpu_affinity.set_affinity(
                args.local_rank,
                nproc_per_node,
                args.affinity
            )
        print(f'{args.local_rank}: thread affinity: {affinity}')

    torch.backends.cudnn.benchmark = True

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    setup_logger(args)

    config = CONFIGS[args.dataset]()
    if args.overwrite_config:
        config.__dict__.update(json.loads(args.overwrite_config))

    train_loader, valid_loader, test_loader = load_dataset(args, config)

    grids = [[(name, value) for value in get_attention_hparam_grid(name)] for name in args.attn_names]
    attn_hparam_grids = list(product(*grids))

    attn_module_classes = {name : make_attn_module_class(name) for name in args.attn_names}

    criterion = QuantileLoss(config).cuda()

    best_val_loss = float('inf')
    is_nan = False
    for attn_hparams in attn_hparam_grids:

        attn_hparams = dict(attn_hparams)
        attention_modules = []
        for name in args.attn_names:
            attn_hparams[name]['dropout_rate'] = config.attn_dropout
            if name == 'imp':
                attn_hparams[name]['input_size'] = config.hidden_size

            attention_modules.append(attn_module_classes[name](**attn_hparams[name]))

        model = AltTemporalFusionTransformer(config, attention_modules).cuda()
        if args.ema_decay:
            model_ema = ModelEma(model, decay=args.ema_decay)

        dllogger.log(step='HPARAMS', data={**vars(args), **vars(config)}, verbosity=1)

        # Run dummy iteration to initialize lazy modules
        dummy_batch = next(iter(train_loader))
        dummy_batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in dummy_batch.items()}
        model(dummy_batch)

        optimizer = FusedAdam(model.parameters(), lr=args.lr)
        if args.distributed_world_size > 1:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        print_once('Model params: {}'.format(sum(p.numel() for p in model.parameters())))
        global_step = 0
        perf_meter = PerformanceMeter(benchmark_mode=not args.disable_benchmark)
        if args.use_amp:
            scaler = amp.GradScaler(init_scale=32768.0)

        validate.best_valid_loss = float('inf')
        validate.early_stop_c = 0

        for epoch in range(args.epochs):
            dllogger.log(step=global_step, data={'epoch': epoch}, verbosity=1)

            model.train()
            for local_step, batch in enumerate(train_loader):
                perf_meter.reset_current_lap()
                batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in batch.items()}
                with torch.jit.fuser("fuser2"), amp.autocast(enabled=args.use_amp):
                    predictions = model(batch)
                    targets = batch['target'][:,config.encoder_length:,:]
                    p_losses = criterion(predictions, targets)
                    loss = p_losses.sum()
                    if loss.isnan().any():
                    	is_nan = True
                    	break
                if global_step == 0 and args.ema_decay:
                    model_ema(batch)
                if args.use_amp:
                    scaler.scale(loss).backward()

                else:
                    loss.backward()
                if not args.grad_accumulation or (global_step+1) % args.grad_accumulation == 0:
                    if args.use_amp:
                        scaler.unscale_(optimizer)
                    if args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    if args.use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    if args.ema_decay:
                        model_ema.update(model)

                if args.distributed_world_size > 1:
                    dist.all_reduce(p_losses)
                    p_losses /= args.distributed_world_size
                    loss = p_losses.sum()

                torch.cuda.synchronize()
                ips = perf_meter.update(args.batch_size * args.distributed_world_size,
                        exclude_from_total=local_step in [0, 1, 2, len(train_loader)-1])

                log_dict = {'P10':p_losses[0].item(), 'P50':p_losses[1].item(), 'P90':p_losses[2].item(), 'loss': loss.item(), 'items/s':ips}
                dllogger.log(step=global_step, data=log_dict, verbosity=1)
                global_step += 1

            if is_nan: 
            	print('NaN encountered')
            	break
            
            validate(args, config, model_ema if args.ema_decay else model, criterion, valid_loader, global_step, attn_hparams)

            if validate.early_stop_c >= args.early_stopping:
                print_once('Early stopping')
                break

        if validate.best_valid_loss < best_val_loss:
            best_val_loss = validate.best_valid_loss
            state_dict = model.module.state_dict() if isinstance(model, (DDP, ModelEma)) else model.state_dict()
            ckpt = {'args': args, 'config': config, 'model': state_dict, 'attn_hparams': attn_hparams}
            torch.save(ckpt, os.path.join(args.results, 'best_model_checkpoint.pt'))

    ### TEST PHASE ###
    state_dict = torch.load(os.path.join(args.results, 'best_model_checkpoint.pt'), map_location='cpu')
    print(state_dict['args'])
    print(state_dict['config'])
    print(state_dict['attn_hparams'])

    attention_modules = [attn_module_classes[name](**attn_hparams[name]) for name in state_dict['args'].attn_names]

    model = AltTemporalFusionTransformer(config, attention_modules).cuda()
    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict['model'])
    model.cuda().eval()

    tgt_scalers = pickle.load(open(os.path.join(args.data_path, 'tgt_scalers.bin'), 'rb'))
    cat_encodings = pickle.load(open(os.path.join(args.data_path,'cat_encodings.bin'), 'rb'))

    unscaled_predictions, unscaled_targets, _, _ = predict(args, config, model, test_loader, tgt_scalers, cat_encodings)

    unscaled_predictions = torch.from_numpy(unscaled_predictions).contiguous()
    unscaled_targets = torch.from_numpy(unscaled_targets).contiguous()

    losses = QuantileLoss(config)(unscaled_predictions, unscaled_targets)
    normalizer = unscaled_targets.abs().mean()
    quantiles = 2 * losses / normalizer

    if args.distributed_world_size > 1:
        quantiles = quantiles.cuda()
        dist.all_reduce(quantiles)
        quantiles /= args.distributed_world_size

    quantiles = {'test_p10': quantiles[0].item(), 'test_p50': quantiles[1].item(), 'test_p90': quantiles[2].item(), 'sum':sum(quantiles).item()}
    finish_log = {**quantiles, 'average_ips':perf_meter.avg, 'convergence_step':validate.conv_step}
    dllogger.log(step=(), data=finish_log, verbosity=1)
    
    finish_log['exp_name'] = args.dataset
    finish_log['attn_name'] = args.attn_name
    finish_log['attn_hparams'] = state_dict['attn_hparams']
    finish_log['config'] = state_dict['config'].__dict__
    
    results_path = os.path.join(args.results, f'results_{args.dataset}_{args.attn_name}.json')
    with open(results_path, 'w') as f:
        json.dump(finish_log, f)
    

def validate(args, config, model, criterion, dataloader, global_step, attn_hparams):
    if not hasattr(validate, 'best_valid_loss'):
        validate.best_valid_loss = float('inf')
    if not hasattr(validate, 'early_stop_c'):
        validate.early_stop_c = 0
    model.eval()

    losses = []
    torch.cuda.synchronize()
    validation_start = time.time()
    for batch in dataloader:
        with torch.jit.fuser("fuser2"), amp.autocast(enabled=args.use_amp), torch.no_grad():
            batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in batch.items()}
            predictions = model(batch)
            targets = batch['target'][:,config.encoder_length:,:]
            p_losses = criterion(predictions, targets)
            bs = next(t for t in batch.values() if t is not None).shape[0]
            losses.append((p_losses, bs))

    torch.cuda.synchronize()
    validation_end = time.time()

    p_losses = sum([l[0]*l[1] for l in losses])/sum([l[1] for l in losses]) #takes into accunt that the last batch is not full
    if args.distributed_world_size > 1:
        dist.all_reduce(p_losses)
        p_losses = p_losses/args.distributed_world_size

    ips = len(dataloader.dataset) / (validation_end - validation_start)

    log_dict = {'P10':p_losses[0].item(), 'P50':p_losses[1].item(), 'P90':p_losses[2].item(), 'loss': p_losses.sum().item(), 'items/s':ips}

    if log_dict['loss'] < validate.best_valid_loss:
        validate.best_valid_loss = log_dict['loss']
        validate.early_stop_c = 0
        validate.conv_step = global_step
        if not dist.is_initialized() or dist.get_rank() == 0:
            state_dict = model.module.state_dict() if isinstance(model, (DDP, ModelEma)) else model.state_dict()
            ckpt = {'args':args, 'config':config, 'model':state_dict, 'attn_hparams': attn_hparams}
            torch.save(ckpt, os.path.join(args.results, 'checkpoint.pt'))
        if args.distributed_world_size > 1:
            dist.barrier()
    else:
        validate.early_stop_c += 1
        
    log_dict = {'val_'+k:v for k,v in log_dict.items()}
    dllogger.log(step=global_step, data=log_dict, verbosity=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=CONFIGS.keys(),
                        help='Dataset name')
    parser.add_argument("--attn_names", type=str, required=True, nargs="*", choices=get_attention_names(),
                        help="List of Attention module names.".format(",".join(get_attention_names())))
    parser.add_argument('--epochs', type=int, default=25,
                        help='Default number of training epochs')
    parser.add_argument('--sample_data', type=lambda x: int(float(x)), nargs=2, default=[-1, -1],
                        help="""Subsample the dataset. Specify number of training and valid examples.
                        Values can be provided in scientific notation. Floats will be truncated.""")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--grad_accumulation', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=1000,
                        help='Stop training if validation loss does not improve for more than this number of epochs.')
    parser.add_argument('--results', type=str, default='/results',
                        help='Directory in which results are stored')
    parser.add_argument('--log_file', type=str, default='dllogger.json',
                        help='Name of dllogger output file')
    parser.add_argument('--overwrite_config', type=str, default='',
                       help='JSON string used to overload config')
    parser.add_argument('--affinity', type=str,
                         default='socket_unique_interleaved',
                         choices=['socket', 'single', 'single_unique',
                                  'socket_unique_interleaved',
                                  'socket_unique_continuous',
                                  'disabled'],
                         help='type of CPU affinity')
    parser.add_argument("--ema_decay", type=float, default=0.0, help='Use exponential moving average')
    parser.add_argument("--disable_benchmark", action='store_true', help='Disable benchmarking mode')

    ARGS = parser.parse_args()
    main(ARGS)
