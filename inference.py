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

import os
import pandas as pd
import numpy as np
import pickle
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modeling import TemporalFusionTransformer
from configuration import ElectricityConfig, make_attn_module_class
from data_utils import TFTDataset
from utils import PerformanceMeter
from criterions import qrisk
import dllogger
from log_helper import setup_logger
from torch.cuda import amp

def _unscale_per_id(config, values, ids, scalers):
    num_horizons = config.example_length - config.encoder_length + 1
    flat_values = pd.DataFrame(
            values,
            columns=[f't{j}' for j in range(num_horizons - values.shape[1], num_horizons)]
            )
    flat_values['id'] = ids
    df_list = []
    for idx, group in flat_values.groupby('id'):
        scaler = scalers[idx]
        group_copy = group.copy()
        for col in group_copy.columns:
            if not 'id' in col:
                _col = np.expand_dims(group_copy[col].values, -1)
                _t_col = scaler.inverse_transform(_col)[:,-1]
                group_copy[col] = _t_col
        df_list.append(group_copy)
    flat_values = pd.concat(df_list, axis=0)

    flat_values = flat_values[[col for col in flat_values if not 'id' in col]]
    return flat_values.values

def _unscale(config, values, scaler):
    num_horizons = config.example_length - config.encoder_length + 1
    flat_values = pd.DataFrame(
            values,
            columns=[f't{j}' for j in range(num_horizons - values.shape[1], num_horizons)]
            )
    for col in flat_values.columns:
        if not 'id' in col:
            _col = np.expand_dims(flat_values[col].values, -1)
            _t_col = scaler.inverse_transform(_col)[:,-1]
            flat_values[col] = _t_col

    flat_values = flat_values[[col for col in flat_values if not 'id' in col]]
    return flat_values.values

def visualize_attn_grids(args, config, key, step, attn_weights):

    for i, attn in enumerate(attn_weights):

        fig, axes = plt.subplots(1, config.n_head, figsize=(config.n_head * 5, 5))
        for j, ax in enumerate(axes):
            ax.grid(False)
            ax.imshow(attn[j])
            ax.set_title(f"attention head {j + 1}")
        plt.tight_layout()

        os.makedirs(os.path.join(args.results, 'attn_grid_vis', str(key[i, 0].item())), exist_ok=True)
        fig.savefig(os.path.join(args.results, 'attn_grid_vis', str(key[i, 0].item()), f'step_{step}_sample_{i}.pdf'))
        plt.close(fig)

def predict(args, config, model, data_loader, scalers, cat_encodings, extend_targets=False, return_attn_vsn_weights=False, visualize_n_attn_weights=0):
    model.eval()
    predictions = []
    targets = []
    ids = []
    perf_meter = PerformanceMeter(benchmark_mode=not args.disable_benchmark)
    n_workers = args.distributed_world_size if hasattr(args, 'distributed_world_size') else 1

    attention_weights = []
    historical_vsn_weights = []
    future_vsn_weights = []
    static_vsn_weights = []

    attn_grid_counter = 0
    with torch.jit.fuser("fuser2"):
        for step, batch in enumerate(data_loader):
            perf_meter.reset_current_lap()
            with torch.no_grad():
                batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in batch.items()}
                ids.append(batch['id'][:,0,:].cpu())
                targets.append(batch['target'].cpu())
                predictions.append(model(batch).float().cpu())

                if visualize_n_attn_weights == -1:
                    visualize_attn_grids(args, config, ids[-1], step, model.attention_weights.float().cpu())
                elif visualize_n_attn_weights > 0 and attn_grid_counter <= visualize_n_attn_weights:
                    visualize_attn_grids(args, config, ids[-1], step, model.attention_weights.float().cpu())
                    attn_grid_counter += 1

                if return_attn_vsn_weights:
                    attention_weights.append(model.attention_weights[:, :, -1].float().cpu())
                    historical_vsn_weights.append(model.historical_vsn_weights.float().cpu()) # (batch_size, enc_length, N) where N = known vars
                    future_vsn_weights.append(model.future_vsn_weights.float().cpu()) # (batch_size, dec_length, M) where M = unknown vars
                    static_vsn_weights.append(model.static_vsn_weights.float().cpu()) # (batch_size, n_statics)

            perf_meter.update(args.batch_size * n_workers,
                exclude_from_total=step in [0, 1, 2, len(data_loader)-1])

    targets = torch.cat(targets, dim=0).cpu().numpy()
    if not extend_targets:
        targets = targets[:,config.encoder_length:,:] 
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    
    if config.scale_per_id:
        ids = torch.cat(ids, dim=0).cpu().numpy()

        unscaled_predictions = np.stack(
                [_unscale_per_id(config, predictions[:,:,i], ids, scalers) for i in range(len(config.quantiles))], 
                axis=-1)
        unscaled_targets = np.expand_dims(_unscale_per_id(config, targets[:,:,0], ids, scalers), axis=-1)
    else:
        ids = None
        unscaled_predictions = np.stack(
                [_unscale(config, predictions[:,:,i], scalers['']) for i in range(len(config.quantiles))], 
                axis=-1)
        unscaled_targets = np.expand_dims(_unscale(config, targets[:,:,0], scalers['']), axis=-1)

    if return_attn_vsn_weights:
        attention_weights = torch.cat(attention_weights, dim=0).cpu().numpy()
        historical_vsn_weights = torch.cat(historical_vsn_weights, dim=0).cpu().numpy()
        future_vsn_weights = torch.cat(future_vsn_weights, dim=0).cpu().numpy()
        static_vsn_weights = torch.cat(static_vsn_weights, dim=0).cpu().numpy()
        return (unscaled_predictions, unscaled_targets, ids, perf_meter,
                attention_weights, historical_vsn_weights, future_vsn_weights, static_vsn_weights)
    else:
        return unscaled_predictions, unscaled_targets, ids, perf_meter

def visualize_v2(args, config, model, data_loader, scalers, cat_encodings):
    (unscaled_predictions, unscaled_targets, ids, _,
     attention_weights, historical_vsn_weights, future_vsn_weights, static_vsn_weights) = predict(args, config, model, data_loader, scalers, cat_encodings,
                                                                                                  extend_targets=True,
                                                                                                  return_attn_vsn_weights=True,
                                                                                                  visualize_n_attn_weights=args.visualize)

    unscaled_predictions, unscaled_targets, ids = torch.Tensor(unscaled_predictions), torch.Tensor(unscaled_targets), torch.Tensor(ids)
    attention_weights = torch.Tensor(attention_weights)
    # historical_vsn_weights = torch.Tensor(historical_vsn_weights)
    # future_vsn_weights = torch.Tensor(future_vsn_weights)
    # static_vsn_weights = torch.Tensor(static_vsn_weights)

    num_horizons = config.example_length - config.encoder_length + 1
    pad = unscaled_predictions.new_full((unscaled_targets.shape[0], unscaled_targets.shape[1] - unscaled_predictions.shape[1], unscaled_predictions.shape[2]), fill_value=float('nan'))
    pad[:,-1,:] = unscaled_targets[:,-num_horizons,:]
    unscaled_predictions = torch.cat((pad, unscaled_predictions), dim=1)

    ids = ids.squeeze()
    joint_graphs = torch.cat([unscaled_targets, unscaled_predictions], dim=2)
    graphs = {i:joint_graphs[ids == i, :, :] for i in set(ids.tolist())}

    attn_graphs = {i: attention_weights[ids == i, :, :] for i in set(ids.tolist())}

    n_samples = None if args.visualize == -1 else args.visualize
    for key, g in graphs.items():
        for i, (ex, attn) in enumerate(zip(g[:n_samples], attn_graphs[key][:n_samples])):
            ex = ex.numpy()
            attn = attn.numpy()
            source_range = range(num_horizons - ex.shape[0] - 1, 0)
            target_range = range(0, num_horizons-1)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

            source = ex[:config.encoder_length, 0]
            target = ex[config.encoder_length:, 0]
            pred = ex[config.encoder_length:, 2]

            ax1.plot(source_range, source, '-r', label="Source: 0: t_0")
            ax1.plot(target_range, target, '-b', label=f"Target: $t_0+1: t_0+{num_horizons-1}$")
            ax1.plot(target_range, pred, '-g', label=f"Prediction: $t_0+1: t_0+{num_horizons-1}$")
            _values = ex[config.encoder_length:, [1, 3]]
            ax1.fill_between(target_range, _values[:,0], _values[:,1], alpha=0.2, color='green', label=f"Prediction quantiles: $t_0+1: t_0+{num_horizons-1}$")
            ax1.axvline(0, linestyle='--', color='k')
            ax1.legend()

            total_range = list(source_range) + list(target_range)
            for j in range(config.n_head):
                ax2.plot(total_range, attn[j], label=f"Attention head {j + 1}")
            ax2.axvline(0, linestyle='--', color='k')
            ax2.legend()

            ax1.set_title("Quantile Prediction")
            ax2.set_title(f"Attention weights for horizon: $0: t_0+{num_horizons-1}$")

            plt.tight_layout()

            os.makedirs(os.path.join(args.results, 'pred_attn_vis', str(key)), exist_ok=True)
            fig.savefig(os.path.join(args.results, 'pred_attn_vis', str(key), f'{i}.pdf'))
            plt.close(fig)

def inference(args, config, model, data_loader, scalers, cat_encodings):
    unscaled_predictions, unscaled_targets, ids, perf_meter = predict(args, config, model, data_loader, scalers, cat_encodings)
    unscaled_predictions, unscaled_targets, ids = torch.Tensor(unscaled_predictions), torch.Tensor(unscaled_targets), torch.Tensor(ids)

    if args.joint_visualization or args.save_predictions:
        ids = ids.squeeze()
        #ids = torch.cat([x['id'][0] for x in data_loader.dataset])
        joint_graphs = torch.cat([unscaled_targets, unscaled_predictions], dim=2)
        graphs = {i:joint_graphs[ids == i, :, :] for i in set(ids.tolist())}
        for key, g in graphs.items(): #timeseries id, joint targets and predictions
            _g = {'targets': g[:,:,0]}
            _g.update({f'P{int(q*100)}':g[:,:,i+1] for i, q in enumerate(config.quantiles)})
            
            if args.joint_visualization:
                summary_writer = SummaryWriter(log_dir=os.path.join(args.results, 'predictions_vis', str(key)))
                for q, t in _g.items(): # target and quantiles, timehorizon values
                    if q == 'targets':
                        targets = torch.cat([t[:,0], t[-1,1:]]) # WIP
                        # We want to plot targets on the same graph as predictions. Probably could be written better.
                        for i, val in enumerate(targets):
                            summary_writer.add_scalars(str(key), {f'{q}':val}, i)
                        continue

                    # Tensor t contains different time horizons which are shifted in phase
                    # Next lines realign them
                    y = t.new_full((t.shape[0] + t.shape[1] -1, t.shape[1]), float('nan'))
                    for i in range(y.shape[1]):
                        y[i:i+t.shape[0], i] = t[:,i]

                    for i, vals in enumerate(y): # timestep, timehorizon values value
                        summary_writer.add_scalars(str(key), {f'{q}_t+{j+1}':v for j,v in enumerate(vals) if v == v}, i)
                summary_writer.close()

            if args.save_predictions:
                for q, t in _g.items():
                    df = pd.DataFrame(t.tolist())
                    df.columns = [f't+{i+1}' for i in range(len(df.columns))]
                    os.makedirs(os.path.join(args.results, 'predictions', str(key)), exist_ok=True)
                    df.to_csv(os.path.join(args.results, 'predictions', str(key), q+'.csv'))

    #losses = QuantileLoss(config)(torch.from_numpy(unscaled_predictions).contiguous(),
    #        torch.from_numpy(unscaled_targets).contiguous()).numpy()
    #normalizer = np.mean(np.abs(unscaled_targets))
    #q_risk = 2 * losses / normalizer
    risk = qrisk(unscaled_predictions.numpy(), unscaled_targets.numpy(), np.array(config.quantiles))

    perf_dict = {
                'throughput': perf_meter.avg,
                'latency_avg': perf_meter.total_time/len(perf_meter.intervals),
                'latency_p90': perf_meter.p(90),
                'latency_p95': perf_meter.p(95),
                'latency_p99': perf_meter.p(99),
                'total_infernece_time': perf_meter.total_time,
                }

    return risk, perf_dict


def main(args):
    
    setup_logger(args)
    # Set up model
    state_dict = torch.load(args.checkpoint)
    config = state_dict['config']
    attn_hparams = state_dict['attn_hparams']
    attn_module_class = make_attn_module_class(state_dict['args'].attn_name)

    attn_module = attn_module_class(**attn_hparams)
    model = TemporalFusionTransformer(config, attn_module).cuda()
    model.load_state_dict(state_dict['model'])
    model.eval()
    model.cuda()

    # Set up dataset
    test_split = TFTDataset(args.data, config)
    data_loader = DataLoader(test_split, batch_size=args.batch_size, num_workers=4)

    scalers = pickle.load(open(args.tgt_scalers, 'rb'))
    cat_encodings = pickle.load(open(args.cat_encodings, 'rb'))

    if args.visualize != 0:
        # TODO: abstract away all forms of visualization.
        visualize_v2(args, config, model, data_loader, scalers, cat_encodings)

    quantiles, perf_dict = inference(args, config, model, data_loader, scalers, cat_encodings)
    quantiles = {'test_p10': quantiles[0].item(), 'test_p50': quantiles[1].item(), 'test_p90': quantiles[2].item(), 'sum':sum(quantiles).item()}
    finish_log = {**quantiles, **perf_dict}
    dllogger.log(step=(), data=finish_log, verbosity=1)
    print('Test q-risk: P10 {test_p10} | P50 {test_p50} | P90 {test_p90}'.format(**quantiles))
    print('Latency:\n\tAverage {:.3f}s\n\tp90 {:.3f}s\n\tp95 {:.3f}s\n\tp99 {:.3f}s'.format(
        perf_dict['latency_avg'], perf_dict['latency_p90'], perf_dict['latency_p95'], perf_dict['latency_p99']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        help='Path to the checkpoint')
    parser.add_argument('--data', type=str,
                        help='Path to the test split of the dataset')
    parser.add_argument('--tgt_scalers', type=str,
                        help='Path to the tgt_scalers.bin file produced by the preprocessing')
    parser.add_argument('--cat_encodings', type=str,
                        help='Path to the cat_encodings.bin file produced by the preprocessing')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--visualize', type=int, default=32,
                        help='Amount of sample predictions to visualize - each example on the separate plot. -1 means all predictions, and 0 means no predictions.')
    parser.add_argument('--joint_visualization', action='store_true',
                        help='Visualize predictions - each timeseries on separate plot. Projections will be concatenated.')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--results', type=str, default='/results')
    parser.add_argument('--log_file', type=str, default='dllogger.json')
    parser.add_argument("--disable_benchmark", action='store_true',
                        help='Disable benchmarking mode')
    ARGS = parser.parse_args()
    main(ARGS)
