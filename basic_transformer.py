"""
This file contains several buidling blocks for Transformer-based architectures. The following building blocks are implemented:
- Positional encoding module (according to the paper "Attention is all you need" (https://arxiv.org/abs/1706.03762))
- Causal 1D-convolution embedding module
- Scaled dot product attention module
- Multi-head attention module (for scaled dot product attention)
- Transformer encoder block
- Transformer decoder block
"""

import torch
from torch import nn
from torch.nn import functional as F

from attention import DotProductAttention

import copy
import math

"""
Standard positional encoding layer adopted from "Attention Is All You Need"
"""
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


"""
Causal 1-dimensional convolutional embedding.
Might be better than the naive linear projection embedding
"""
class CausalConv1dEmbedding(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, **kwargs
        )

    def forward(self, x):
        return F.leaky_relu(self.conv(F.pad(x, pad=(self.padding, 0))))


"""
Multi-head attention mechanism.
"""
class MultiHeadAttention(nn.Module):

    def __init__(
        self, attention: nn.Module, n_heads: int, n_hidden: int, n_out: int, bias=False
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.W_q = nn.LazyLinear(n_hidden, bias=bias)
        self.W_k = nn.LazyLinear(n_hidden, bias=bias)
        self.W_v = nn.LazyLinear(n_hidden, bias=bias)
        self.W_o = nn.LazyLinear(n_out)
        self.attention = attention
        self.attn_weights = torch.Tensor(0)

    def transpose_QKV(self, X: torch.Tensor):
        X = X.reshape(*X.shape[:2], self.n_heads, -1)
        X = X.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, n_hidden/n_heads)
        return X

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        K = self.transpose_QKV(self.W_k(keys))
        V = self.transpose_QKV(self.W_v(values))
        Q = self.transpose_QKV(self.W_q(queries))
        # Q, K, V: (batch_size, n_heads, seq_len, n_hidden/n_heads)

        out, self.attn_weights = self.attention(Q, K, V, mask)
        out = out.reshape(
            out.shape[0], out.shape[2], -1
        )  # (batch_size, seq_len, n_hidden*n_heads)

        return self.W_o(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        n_heads=8,
        n_hidden=64,
        n_out=512,
        ffn_n_hidden=2048,
        _attention=DotProductAttention(),
        dropout=0.1,
        norm_first=True,
    ):
        """
        :param n_heads: number of attention heads
        :param n_hidden: dimensionality of each attention head
        :param n_out: dimensionality of output (after multi-head attention and after point-wise feedforward network)
        :param ffn_n_hidden: hidden dimension of feedforward network
        :param _attention: self attention module (default: DotProductAttention)
        :param dropout: dropout rate
        :param norm_first: whether to apply layer normalization before attention layer or after
        """
        super().__init__()
        self.norm_first = norm_first
        self.mha = MultiHeadAttention(_attention, n_heads, n_hidden, n_out)
        self.norm1 = nn.LayerNorm(n_out)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LazyLinear(ffn_n_hidden), nn.ReLU(), nn.LazyLinear(n_out)
        )
        self.norm2 = nn.LayerNorm(n_out)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None):
        if self.norm_first:
            X = self.norm1(X)
            X = X + self.mha(X, X, X, mask)
            X = X + self.ffn(self.norm2(X))
        else:
            X = self.norm1(X + self.mha(X, X, X, mask))
            X = self.norm2(X + self.ffn(X))

        return self.dropout(X)


"""
A single Transformer decoder block which contains:
1. Multi-head attention layer
2. Point-wise Feedforward network layer
with residual connections and layer normalization
"""
class TransformerDecoderBlock(nn.Module):

    def __init__(
        self,
        n_heads=8,
        n_hidden=64,
        n_out=512,
        ffn_n_hidden=2048,
        _self_attention=DotProductAttention(),
        _cross_attention=DotProductAttention(),
        dropout=0.1,
        norm_first=True,
    ):
        """
        :param n_heads: number of attention heads
        :param n_hidden: dimensionality of each attention head
        :param n_out: dimensionality of output (after multi-head attention and after point-wise feedforward network)
        :param ffn_n_hidden: hidden dimension of feedforward network
        :param _self_attention: self attention module (default: DotProductAttention)
        :param _cross_attention: cross attention module (default: DotProductAttention)
        :param dropout: dropout rate
        :param norm_first: whether to apply layer normalization before attention layer or after
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.mha1 = MultiHeadAttention(_self_attention, n_heads, n_hidden, n_out)
        self.norm1 = nn.LayerNorm(n_out)

        self.mha2 = MultiHeadAttention(_cross_attention, n_heads, n_hidden, n_out)
        self.norm2 = nn.LayerNorm(n_out)

        self.ffn = nn.Sequential(
            nn.LazyLinear(ffn_n_hidden), nn.ReLU(), nn.LazyLinear(n_out)
        )
        self.norm3 = nn.LayerNorm(n_out)

    def forward(
        self, X: torch.Tensor, enc_outputs: torch.Tensor, mask: torch.Tensor = None
    ):
        # X: (batch_size, seq_len, emb)
        # enc_outputs: (batch_size, seq_len, emb)
        # mask: (seq_len, seq_len)
        if self.norm_first:
            X = self.norm1(X)
            X = X + self.mha1(X, X, X, mask)
            X = self.norm2(X)
            X = X + self.mha2(X, enc_outputs, enc_outputs)
            X = X + self.ffn(self.norm3(X))
        else:
            X = self.norm1(X + self.mha1(X, X, X, mask))
            X = self.norm2(X + self.mha2(X, enc_outputs, enc_outputs))
            X = self.norm3(X + self.ffn(X))

        return self.dropout(X)


"""
Base class for the autoregressive decoder-only Transformer model. 
Uses naive linear layer for embedding.
"""
class BaseDecoderOnlyTransformer(nn.Module):

    def __init__(
        self,
        d_in=2,
        emb_size=512,
        n_heads=8,
        n_hidden=64,
        ffn_n_hidden=2048,
        num_layers=3,
        _attention=DotProductAttention(),
        norm_first=True,
    ):
        """
        :param d_in: total number of input features (N timeseries + M covariates)
        :param emb_size: embedding size of d_in (d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param _attention: attention module (default: DotProductAttention)
        :param norm_first: whether to apply layer normalization before or after attention
        """
        super().__init__()
        self.emb = nn.Linear(d_in, emb_size)
        self.pos_enc = PositionalEncoding(emb_size)

        decoder_block = TransformerEncoderBlock(
            n_heads=n_heads,
            n_hidden=n_hidden,
            n_out=emb_size,
            ffn_n_hidden=ffn_n_hidden,
            _attention=_attention,
            norm_first=norm_first,
        )
        self.transformer_blocks = nn.ModuleList(
            [copy.deepcopy(decoder_block) for _ in range(num_layers)]
        )

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        # covariates X:  (batch_size, seq_len, cov_d)
        # features fX:   (batch_size, seq_len, feat_d)

        # (batch_size, seq_len, cov_d + feat_d)
        Y = torch.cat([X, fX], dim=-1)

        # embedding: (batch_size, seq_len, cov_d + feat_d) -> (batch_size, seq_len, emb_size)
        # positional embedding: (seq_len) -> (seq_len, emb_size)
        Y = self.pos_enc(self.emb(Y))

        # through decoder blocks with same mask shape (seq_len, seq_len)
        for block in self.transformer_blocks:
            Y = block(Y, mask=mask)

        # output shape: (batch_size, seq_len, emb_size)
        return Y


"""
Decoder-only Transformer model that predicts d_out values for each step.
"""
class QuantileTransformer(BaseDecoderOnlyTransformer):
    def __init__(
        self,
        d_in=2,
        n_quantiles=3,
        emb_size=512,
        n_heads=8,
        n_hidden=64,
        ffn_n_hidden=2048,
        num_layers=3,
        _attention=DotProductAttention(),
        norm_first=True
    ):
        """
        :param d_in: total number of input features (N timeseries + M covariates)
        :param n_quantiles: number of output quantiles (N timeseries)
        :param emb_size: embedding size of d_in (d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param _attention: attention module (default: DotProductAttention)
        :param norm_first: whether to apply layer normalization before or after attention
        """
        super().__init__(
            d_in,
            emb_size,
            n_heads,
            n_hidden,
            ffn_n_hidden,
            num_layers,
            _attention,
            norm_first,
        )
        self.fc = nn.Linear(emb_size, n_quantiles)

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        Y = super().forward(X, fX, mask)
        # dense layer to project to n_quantiles dims
        # (batch_size, seq_len, emb_size) -> (batch_size, seq_len, n_quantiles)
        return self.fc(Y)


"""
Base class for the autoregressive decoder-only Transformer model. 
Uses convolutional layer for the embedding.
"""
class BaseConvDecoderOnlyTransformer(nn.Module):

    def __init__(
        self,
        d_in=2,
        emb_size=512,
        n_heads=8,
        n_hidden=64,
        ffn_n_hidden=2048,
        num_layers=3,
        norm_first=True,
        _attention=DotProductAttention(),
        conv_kernel_size=3,
        conv_dilation=1,
    ):
        """
        :param d_in: total number of input features (N timeseries + M covariates)
        :param emb_size: embedding size of d_in (d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param norm_first: whether to apply layer normalization before or after attention
        :param _attention: attention module (default: DotProductAttention)
        :param conv_kernel_size: kernel size of the convolutional embedding layer
        :param conv_dilation: dilation of the convolutional embedding layer
        """
        super().__init__()
        self.emb = CausalConv1dEmbedding(
            d_in, emb_size, conv_kernel_size, conv_dilation
        )
        self.pos_enc = PositionalEncoding(emb_size)

        decoder_block = TransformerEncoderBlock(
            n_heads=n_heads,
            n_hidden=n_hidden,
            n_out=emb_size,
            ffn_n_hidden=ffn_n_hidden,
            _attention=_attention,
            norm_first=norm_first,
        )
        self.transformer_blocks = nn.ModuleList(
            [copy.deepcopy(decoder_block) for _ in range(num_layers)]
        )

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        # covariates X:  (batch_size, seq_len, cov_d)
        # features fX:   (batch_size, seq_len, feat_d)

        # (batch_size, seq_len, cov_d + feat_d)
        Y = torch.cat([X, fX], dim=-1)

        Y = Y.transpose(1, 2)

        # embedding: (batch_size, cov_d + feat_d, seq_len) -> (batch_size, emb_size, seq_len)
        Y = self.emb(Y)

        # positional embedding: (seq_len) -> (seq_len, emb_size)
        Y = self.pos_enc(Y.transpose(1, 2))

        # through decoder blocks with same mask shape (seq_len, seq_len)
        for block in self.transformer_blocks:
            Y = block(Y, mask=mask)

        # output shape: (batch_size, seq_len, emb_size)
        return Y


"""
Decoder-only Transformer model with convolutional embedding that predicts d_out values for each step.
"""
class QuantileConvDecoderOnlyTransformer(BaseConvDecoderOnlyTransformer):
    def __init__(
        self,
        d_in=2,
        n_quantiles=3,
        emb_size=512,
        n_heads=8,
        n_hidden=64,
        ffn_n_hidden=2048,
        num_layers=3,
        norm_first=True,
        _attention=DotProductAttention(),
        conv_kernel_size=3,
        conv_dilation=1,
    ):
        """
        :param d_in: total number of input features (N timeseries + M covariates)
        :param n_quantiles: number of output features (N timeseries)
        :param emb_size: embedding size of d_in (d_model)
        :param n_heads: number of heads in multi-head attention
        :param n_hidden: number of units in Query, Key, Value projection in multi-head attention
        :param ffn_n_hidden: number of hidden units in point-wise FFN
        :param num_layers: number of attention decoder layers
        :param norm_first: whether to apply layer normalization before or after attention
        :param _attention: attention module (default: DotProductAttention)
        :param conv_kernel_size kernel size of the convolutional embedding layer
        :param conv_dilation dilation of the convolutional embedding layer
        """
        super().__init__(
            d_in,
            emb_size,
            n_heads,
            n_hidden,
            ffn_n_hidden,
            num_layers,
            norm_first,
            _attention,
            conv_kernel_size,
            conv_dilation,
        )
        self.fc = nn.Linear(emb_size, n_quantiles)

    def forward(self, X: torch.Tensor, fX: torch.Tensor, mask: torch.Tensor = None):
        Y = super().forward(X, fX, mask)
        # dense layer to project to n_quantiles dims
        # (batch_size, seq_len, emb_size) -> (batch_size, seq_len, n_quantiles)
        return self.fc(Y)