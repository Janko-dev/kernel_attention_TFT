import copy
import torch
import matplotlib.pyplot as plt

from basic_transformer import QuantileTransformer, QuantileConvDecoderOnlyTransformer
from criterions import QuantileLoss


def create_model(attention_module, config):
    return QuantileTransformer(
        d_in=2,
        n_quantiles=3,
        emb_size=config.emb_size,
        n_heads=config.n_heads,
        n_hidden=config.n_hidden,
        ffn_n_hidden=config.ffn_n_hidden,
        num_layers=1,
        _attention=attention_module,
        norm_first=True
    )


def train_step(model, criterion, train_dataloader, optimizer, mask, device):
    model.train()
    train_loss = torch.zeros_like(criterion.q)
    n = 0
    mask = mask.to(device)
    for sample in train_dataloader:
        src_X, src_fX, tgt_X, tgt_fX = (v.to(device) for v in sample)

        out = model(src_X, src_fX, mask)
        p_loss = criterion(out[:, -tgt_X.shape[1]:], tgt_fX)

        optimizer.zero_grad()
        p_loss.sum().backward()
        optimizer.step()

        train_loss += p_loss * src_X.shape[0]
        n += src_X.shape[0]

    return train_loss / n


def val_step(model, criterion, val_dataloader, mask, device):
    model.eval()
    val_loss = torch.zeros_like(criterion.q)
    n = 0
    mask = mask.to(device)
    with torch.no_grad():
        for sample in val_dataloader:
            src_X, src_fX, tgt_X, tgt_fX = (v.to(device) for v in sample)

            out = model(src_X, src_fX, mask)
            p_loss = criterion(out[:, -tgt_X.shape[1]:], tgt_fX)

            val_loss += p_loss * src_X.shape[0]
            n += src_X.shape[0]

    return val_loss / n


def format_print(loss: dict):
    return ', '.join([f"{p}:{l:.4f}" for p, l in loss.items()])


def train(attn_module, config, train_dl, val_dl, mask, verbose=False):
    model = create_model(attn_module, config).to(config.device)

    train_history = []
    val_history = []

    criterion = QuantileLoss(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    best_val_loss = float('inf')
    early_stopping_c = 0
    best_model = None

    for epoch in range(config.epochs):

        train_loss = train_step(model, criterion, train_dl, optimizer, mask, config.device)
        train_history.append({'p10': train_loss[0].item(), 'p50': train_loss[1].item(), 'p90': train_loss[2].item()})

        val_loss = val_step(model, criterion, val_dl, mask, config.device)
        val_history.append({'p10': val_loss[0].item(), 'p50': val_loss[1].item(), 'p90': val_loss[2].item()})

        if verbose: print(f"{epoch=}\t{format_print(train_history[-1])}\t{format_print(val_history[-1])}")

        val_loss_sum = val_loss.sum().item()
        scheduler.step(val_loss_sum)

        if val_loss_sum < best_val_loss:
            best_val_loss = val_loss_sum
            early_stopping_c = 0
            best_model = copy.deepcopy(model)
        else:
            early_stopping_c += 1

        if early_stopping_c > config.patience: break

    return best_model, train_history, val_history, best_val_loss


def inference(idx, best_model, test_set, config, mask, attn_name):
    with torch.no_grad():
        src_X, src_fX, tgt_X, tgt_fX = test_set[idx]

        out = best_model(src_X.unsqueeze(0).to(config.device),
                         src_fX.unsqueeze(0).to(config.device),
                         mask.to(config.device))

        pred = out[:, -tgt_X.shape[0]:].squeeze().cpu()

        num_horizons = tgt_X.shape[0]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        ax1.plot(src_X, src_fX, '-r', label="Source: 0: t_0")
        ax1.plot(tgt_X, tgt_fX, '-b', label=f"Target: $t_0+1: t_0+{num_horizons}$")
        ax1.plot(tgt_X, pred[:, 1], '-g', label=f"Prediction: $t_0+1: t_0+{num_horizons}$")
        ax1.fill_between(tgt_X.squeeze(), pred[:, 0], pred[:, 2], alpha=0.2, color='green',
                         label=f"Prediction quantiles: $t_0+1: t_0+{num_horizons}$")
        ax1.axvline(src_X[-1, 0], linestyle='--', color='k')
        ax1.legend()

        attn_w = best_model.transformer_blocks[0].mha.attn_weights[0].cpu()
        for j, attn in enumerate(attn_w):
            ax2.plot(src_X, attn[-1], label=f"Attention head {j + 1}")
        ax2.axvline(src_X[-1, 0], linestyle='--', color='k')
        ax2.legend()

        ax1.set_title(f"Quantile Prediction for attention module: {attn_name}")
        ax2.set_title(f"Attention weights for horizon: $0: t_0+{num_horizons}$")

        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, attn_w.shape[0], figsize=(5 * attn_w.shape[0], 5))
        fig.suptitle(f'Attention weight matrix visualization for module: {attn_name}')

        for j, (ax, attn) in enumerate(zip(axes, attn_w)):
            t = torch.arange(attn.shape[0]).reshape(-1, 1)
            attn = attn.cpu() * t
            ax.imshow(attn)
            ax.set_title(f'Attention head {j + 1}')

        plt.show()

