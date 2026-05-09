# -*- coding: utf-8 -*-

#@title Installs

#@title Imports
# for HyenaDNA specifically
import math
from collections import namedtuple
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch import Tensor
from torchvision.ops import StochasticDepth

"""# HyenaDNA


"""

#@title Hyena layer


def fftconv(u, k, D):
    """Apply a convolution in the Fourier domain."""
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    """The Sin activation function for the Hyena Filter function."""
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float=1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # Normalize the time embedding so the filter sees a fixed range.
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    """The window function applied to the output of the (MLP) filter function."""
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool=True,
        shift: float = 0.05,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(
            self,
            d_model,
            emb_dim=3, # dim of input to MLP, augments with positional encoding
            order=16, # width of the implicit MLP
            fused_fft_conv=False,
            seq_len=1024,
            lr=1e-3,
            lr_pos_emb=1e-5,
            dropout=0.0,
            w=1, # frequency of periodic activations
            wd=0, # weight decay of kernel parameters
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **kwargs
        ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()

        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y


class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2,
            filter_order=64,
            dropout=0.0,
            filter_dropout=0.0,
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args
        )

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')

        uc = self.short_filter(u)[...,:l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y

#@title Self-Attention (alternative)

"""
If you'd like to try the HyenaDNA model using attention instead, you can. ie,
use a regular decoder only Transformer.

Borrowed from the FlashAttention library by Tri Dao.
"""

class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype,
                                      device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, 'b s -> b 1 1 s')
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
        return output

class MHA(nn.Module):
    """Multi-head self-attention and cross-attention
    """

    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0,
                 softmax_scale=None, causal=False, layer_idx=None, dwconv=False,return_residual=False,device=None, dtype=None) -> None:
        """
            return_residual: whether to return the input x along with the output. This is for
                performance reason: for post-norm architecture, returning the input allows us
                to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.return_residual = return_residual

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        linear_cls = nn.Linear
        linear_resid_cls = LinearResidual
        inner_attn_cls =  SelfAttention

        if not self.return_residual:
            self.Wqkv = linear_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        else:
            self.Wqkv = linear_resid_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        if self.dwconv:
            self.dwconv_qkv = nn.Conv1d(3 * embed_dim, 3 * embed_dim, kernel_size=3, padding=2,
                                        groups=3 * embed_dim)

        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                         attention_dropout=dropout)

        # output projection always have the bias (for now)
        self.out_proj = linear_cls(embed_dim, embed_dim, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """

        kwargs = ({'key_padding_mask': key_padding_mask, **kwargs})

        if not self.return_residual:
            qkv = self.Wqkv(x)
        else:
            qkv, x = self.Wqkv(x)
        if self.dwconv:
            qkv = rearrange(self.dwconv_qkv(rearrange(qkv, 'b s d -> b d s'))[..., :-2],
                            'b d s -> b s d').contiguous()
        qkv = rearrange(qkv, '... (three h d) -> ... three h d', three=3, d=self.head_dim)

        context = self.inner_attn(qkv, **kwargs)

        out = self.out_proj(rearrange(context, '... h d -> ... (h d)'))
        return out if not self.return_residual else (out, x)

#@title MLP layer

"""
The MLP layer after the mixer layer (HyenaOperator).
"""

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu,
                 return_residual=False, device=None, dtype=None):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/mlp.py
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

#@title Block layer (Hyena + MLP layers)

"""
A block consists of a Mixer layer (Hyena or attention), and a MLP layer.

"""

class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input

class Block(nn.Module):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0.,
                 return_residual=False,
                 residual_in_fp32=False):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/block.py
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls()
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual = None,
                mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out

            hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out))
                                        + hidden_states).to(dtype=self.norm1.weight.dtype))

            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out

                hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out))
                                            + hidden_states).to(dtype=self.norm2.weight.dtype))

            return hidden_states

def create_mixer_cls(layer=None,
                     attn_layer_idx=None, attn_cfg=None, layer_idx=None,
                     device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop('causal', True)

        mha_cls = MHA

        mixer_cls = partial(mha_cls, causal=causal, layer_idx=layer_idx,
                            **(attn_cfg if attn_cfg is not None else {}),**factory_kwargs)
    else:
        # mixer_cls = instantiate(registry.layer, layer, partial=True, layer_idx=layer_idx, **factory_kwargs)

        mixer_cls = partial(HyenaOperator, **layer)

    return mixer_cls

def create_mlp_cls(d_model, d_inner=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model

    mlp_cls = partial(Mlp, hidden_features=inner_dim,
                          activation=partial(F.gelu, approximate='tanh'), **factory_kwargs)

    return mlp_cls


def create_block(d_model, d_inner=None,
                 layer=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False,
                 layer_idx=None,
                 device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mixer_cls = create_mixer_cls(layer=layer,
                                 attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg, layer_idx=layer_idx,
                                 **factory_kwargs)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner,
                             **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2,residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)

#@title Backbone model (stack of blocks)

"""
A backbone model consists of a stack of blocks. If you use attention, then
positional embeddings are included. When using Hyena, then the pos emb
revert to doing nothing.
"""

class GPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None,
                 word_embed_proj_dim=None, device=None, dtype=None):
        """
            If max_position_embeddings <= 0, there's no position embeddings
            If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
                the project up to embed_dim
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx,
                                                **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim,
                                                padding_idx=padding_idx, **factory_kwargs)
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False,
                                        **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim,
                                                    **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        """
            input_ids: (batch, seqlen)
            position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

class LMBackbone(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.process_group = process_group
        self.residual_in_fp32 = residual_in_fp32
        # note max_position_embeddings is 0 for Hyena, and therefore isn't used
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings,
                                             **factory_kwargs)

        self.layers = nn.ModuleList([create_block(
            d_model, d_inner=d_inner,
            layer=layer, attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg, layer_norm_epsilon=layer_norm_epsilon,
            resid_dropout1=embed_dropout if i == 0 else resid_dropout,
            resid_dropout2=resid_dropout, residual_in_fp32=residual_in_fp32,layer_idx=i,
            **factory_kwargs,
        ) for i in range(n_layer)])

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)

        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids,)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

        return hidden_states

#@title Decoder head layer

"""
A simple decoder head (using MLP) to predict a sequence level classification.
You have the option to average across all the tokens in a sequence or using the
"last" token to classify.  At least, those 2 worked best for us, but we provide
other "modes" as well.

We only need this for classification.  Otherwise we'll use the hidden
states of the backbone as embeddings.

"""


class SequenceDecoder(nn.Module):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last"
    ):
        super().__init__()

        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool":
            restrict = lambda x: (
                torch.cumsum(x, dim=-2)
                / torch.arange(
                    1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
            )[..., -l_output:, :]

            def restrict(x):
                L = x.size(-2)
                s = x.sum(dim=-2, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x[..., -(l_output - 1) :, :].flip(-2), dim=-2)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(-2)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x.dtype, device=x.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        return self.output_transform(x)

#@title Model (backbone + head)

"""
Putting it all together, the model consists of a backbone model
and a decoder head (you can turn off head for embeddings only too).

Here we use a simple head to do multi-classification, but
can also swap the head to do next token prediction too.  We defer to the main
HyenaDNA for that code, since pretraining with next token prediction isn't quite
feasible on colab.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F



from utils.model_head import UNet1D
from utils.feature_emb import LinearReLUEmbeddings
class HyenaDNAModel(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 layer=None, attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1, use_head=False, n_classes: int = 2,
                 device=None, dtype=None, stage: str = "train", **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.stage = stage
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.use_head = use_head
        self.head=UNet1D()

        # check if layer (config) has d_model (HF code differs from main Safari code)
        if 'd_model' not in layer:
            layer['d_model'] = d_model


        self.backbone = LMBackbone(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
            layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout, embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg, residual_in_fp32=residual_in_fp32,
            **factory_kwargs, **kwargs
        )



        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

     

    def forward(self, input_ids, pileup_embed, position_ids=None, state=None): # state for the repo interface
        hidden_states = self.backbone(input_ids, position_ids=position_ids)

        d_feature_emb=128
        embeddings = LinearReLUEmbeddings(10000,2, d_feature_emb)
        model_feature_emb = nn.Sequential(embeddings,nn.Flatten(2))
        model_feature_emb.to(hidden_states.device)

        if self.stage != 'test':
            upsample_factor=torch.FloatTensor(pileup_embed.shape[0]).uniform_(1, 3).resize_(pileup_embed.shape[0],1,1).to(hidden_states.device)
            emb_cat=torch.cat((hidden_states, model_feature_emb(torch.log1p(torch.round(pileup_embed[:,:,[0,1]] * upsample_factor))) ),dim=2)
        else:
            emb_cat=torch.cat((hidden_states, model_feature_emb(torch.log1p(torch.round(pileup_embed[:,:,[0,1]] ))) ),dim=2)

        return self.head(emb_cat)
    

#@title Huggingface Pretrained Wrapper
# for Huggingface integration, we use a wrapper class around the model
# to load weights
import json
import os
import subprocess
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
import re

def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict.

    scratch_dict: dict, a state dict from a newly initialized HyenaDNA model
    pretrained_dict: dict, a state dict from the pretrained ckpt
    checkpointing: bool, whether the gradient checkpoint flag was used in the
    pretrained model ckpt. This slightly changes state dict keys, so we patch
    that if used.

    return:
    dict, a state dict with the pretrained weights loaded (head is scratch)

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    """

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes, )  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model

"""# Data pipeline


"""

#@title Tokenizer

"""
Just a simple character level tokenizer.

From: https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py

CharacterTokenzier for Hugging Face Transformers.
This is heavily inspired from CanineTokenizer in transformers package.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, padding_side: str='left', **kwargs):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "N":4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)


from random import random
import csv
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded


# helper functions
def exists(val):
    """Return True when a value is not None."""
    return val is not None

def coin_flip():
    """Sample a fair Bernoulli outcome."""
    return random() > 0.5


string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
# augmentation
def string_reverse_complement(seq):
    """Compute the reverse complement of a nucleotide sequence."""
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp



import torch.optim as optim

import json
import os
import subprocess
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig


import time, pathlib, torch, torchvision, numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)



# ----------------------------------------------------------------------
#  The evaluate() function – exactly what you supplied
# ----------------------------------------------------------------------
def evaluate(model, loader, criterion, device, stage, csv_path=None, return_rows=False):
    model.eval()
    t_total_start = time.perf_counter()
    t_infer = 0.0
    t_concat = 0.0
    t_filter = 0.0
    t_write = 0.0
    t_metrics = 0.0
    total_samples = 0
    total_bw_samples = 0
    total_bw_accuracy = 0
    total_bw_f1 = 0
    total_bw_precision = 0
    total_bw_recall = 0
    total_bw_mcc = 0
    total_bw_auc = 0
    total_tn = total_fp = total_fn = total_tp = 0

    merge_window = 20
    flanking_bp = 16
    if_sortscore = False

    rows_written = 0
    csv_file = None
    csv_writer = None
    csv_path_obj = None
    csv_buffer = []  # Buffer for batch CSV writes
    csv_buffer_size = 128
    predictions_batches = []
    pileup_batches = []
    inputs_batches = []
    actuals_batches = []
    batch_pos_indices_list = []
    batch_sizes = []
    target_ids_batches = []
    start_pos_batches = []
    sample_indices_batches = []
    rows_out = []

    def _to_scalar(value):
        if isinstance(value, str):
            return value
        if torch.is_tensor(value):
            value = value.detach().cpu().view(-1)
            return value[0].item() if value.numel() > 0 else None
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            return flat[0].item() if flat.size > 0 else None
        if isinstance(value, (list, tuple)) and value:
            return _to_scalar(value[0])
        return value

    def flush_csv_buffer():
        """Write buffered rows to CSV."""
        nonlocal csv_buffer
        if csv_buffer and csv_writer:
            csv_writer.writerows(csv_buffer)
            csv_buffer = []

    try:
        if stage == 'test' and csv_path:
            csv_path_obj = Path(csv_path)
            csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path_obj.open("w", newline="")
            csv_writer = csv.writer(csv_file, delimiter="\t")

        with torch.inference_mode():
            t_infer_start = time.perf_counter()
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    sample_indices, inputs, targets, pileup_embed = batch
                else:
                    sample_indices = None
                    inputs, targets, pileup_embed = batch
                if stage != 'test':
                    inputs, targets, pileup_embed = (
                        inputs.to(device),
                        targets.to(device),
                        pileup_embed.to(device),
                    )
                    outputs = model(inputs, pileup_embed=pileup_embed)
                    outputs, targets = outputs.squeeze(), targets.squeeze()
                    actuals = targets.detach().cpu().numpy()
                else:
                    inputs, pileup_embed = inputs.to(device), pileup_embed.to(device)
                    outputs = model(inputs, pileup_embed=pileup_embed).squeeze()

                predictions = torch.sigmoid(outputs).detach().cpu().numpy()
                binary_predictions = predictions >= 0.5
                batch_pos_indices = np.where(binary_predictions == 1)

                # Move tensors to CPU once per batch for efficient array comparisons.
                pileup_embed_cpu = pileup_embed.cpu().numpy() if torch.is_tensor(pileup_embed) else pileup_embed
                inputs_cpu = inputs.cpu().numpy() if torch.is_tensor(inputs) else inputs
                targets_cpu = targets.cpu().numpy() if torch.is_tensor(targets) else targets
                predictions_batches.append(predictions)
                batch_pos_indices_list.append(batch_pos_indices)
                batch_sizes.append(predictions.shape[0])
                pileup_batches.append(pileup_embed_cpu)
                inputs_batches.append(inputs_cpu)
                if stage == 'test':
                    target_ids_batches.append(np.asarray(targets_cpu[0]))
                    start_pos_batches.append(np.asarray(targets_cpu[1]))
                else:
                    actuals_batches.append(actuals)
                if sample_indices is not None:
                    if torch.is_tensor(sample_indices):
                        sample_indices_batches.append(
                            sample_indices.detach().cpu().numpy()
                        )
                    else:
                        sample_indices_batches.append(np.asarray(sample_indices))
            t_infer += time.perf_counter() - t_infer_start

            if predictions_batches:
                t_concat_start = time.perf_counter()
                predictions_all = np.concatenate(predictions_batches, axis=0)
                pileup_embed_all = np.concatenate(pileup_batches, axis=0)
                inputs_all = np.concatenate(inputs_batches, axis=0)
                if sample_indices_batches:
                    sample_indices_all = np.concatenate(sample_indices_batches, axis=0)
                else:
                    sample_indices_all = np.arange(predictions_all.shape[0])
                t_concat += time.perf_counter() - t_concat_start

                t_filter_start = time.perf_counter()
                binary_predictions_all = predictions_all >= 0.5
                pos_batch_indices, pos_indices = np.where(binary_predictions_all == 1)
                keep_mask = (
                    np.ones(pos_batch_indices.shape[0], dtype=bool)
                    if stage == 'test'
                    else None
                )

                # filter predictions based on pile-up (all batches together)
                if pos_batch_indices.size > 0:
                    batch_index = pos_batch_indices
                    pos_index = pos_indices
                    first2 = pileup_embed_all[batch_index, pos_index - 1, :2]
                    zero_first2 = np.all(first2 == 0, axis=1)

                    window_start = pos_index - 1 - flanking_bp
                    window_end = pos_index + flanking_bp
                    safe_window = (
                        (window_start >= 0)
                        & (window_end <= pileup_embed_all.shape[1])
                    )

                    chan0_nonzero = (pileup_embed_all[:, :, 0] != 0).astype(np.int32)
                    prefix = np.cumsum(chan0_nonzero, axis=1)

                    zero_window = np.zeros(pos_batch_indices.shape[0], dtype=bool)
                    if np.any(safe_window):
                        safe_idx = np.where(safe_window)[0]
                        safe_batch = batch_index[safe_idx]
                        end_idx = window_end[safe_idx] - 1
                        start_idx = window_start[safe_idx] - 1
                        sum_end = prefix[safe_batch, end_idx]
                        sum_start = np.where(
                            start_idx >= 0,
                            prefix[safe_batch, start_idx],
                            0,
                        )
                        zero_window[safe_idx] = (sum_end - sum_start) == 0

                    if np.any(~safe_window):
                        unsafe_idx = np.where(~safe_window)[0]
                        for i in unsafe_idx:
                            b = batch_index[i]
                            p = pos_index[i]
                            output_pileup_embed = pileup_embed_all[b]
                            if np.array_equal(
                                output_pileup_embed[
                                    p - 1 - flanking_bp : p + flanking_bp, 0
                                ],
                                [0] * (flanking_bp * 2 + 1),
                            ):
                                zero_window[i] = True

                    invalid_mask = zero_first2 | zero_window
                    if stage == 'test':
                        keep_mask[invalid_mask] = False
                    else:
                        predictions_all[
                            batch_index[invalid_mask], pos_index[invalid_mask]
                        ] = 0
                t_filter += time.perf_counter() - t_filter_start

                if stage == 'test':
                    t_write_start = time.perf_counter()
                    target_ids_all = np.concatenate(target_ids_batches, axis=0)
                    start_pos_all = np.concatenate(start_pos_batches, axis=0)
                    valid_indices = np.where(keep_mask)[0]
                    if valid_indices.size > 0:
                        batch_index = pos_batch_indices[valid_indices]
                        pos_index = pos_indices[valid_indices]
                        seq_len = inputs_all.shape[1]

                        has_left = pos_index - 1 >= 0
                        has_right = pos_index + 1 < seq_len

                        left0 = np.zeros(valid_indices.shape[0], dtype=inputs_all.dtype)
                        left1 = np.zeros(valid_indices.shape[0], dtype=inputs_all.dtype)
                        right0 = np.zeros(valid_indices.shape[0], dtype=inputs_all.dtype)
                        right1 = np.zeros(valid_indices.shape[0], dtype=inputs_all.dtype)

                        if np.any(has_left):
                            left_idx = np.where(has_left)[0]
                            left0[left_idx] = inputs_all[
                                batch_index[left_idx], pos_index[left_idx] - 1
                            ]
                            left1[left_idx] = inputs_all[
                                batch_index[left_idx], pos_index[left_idx]
                            ]

                        if np.any(has_right):
                            right_idx = np.where(has_right)[0]
                            right0[right_idx] = inputs_all[
                                batch_index[right_idx], pos_index[right_idx]
                            ]
                            right1[right_idx] = inputs_all[
                                batch_index[right_idx], pos_index[right_idx] + 1
                            ]

                        is_ac = has_left & (left0 == 7) & (left1 == 8)
                        is_ct = has_right & (right0 == 8) & (right1 == 10)
                        is_ag = has_left & (left0 == 7) & (left1 == 9)
                        is_gt = has_right & (right0 == 9) & (right1 == 10)

                        if_end = np.full(valid_indices.shape[0], -1, dtype=int)
                        strand = np.full(valid_indices.shape[0], -1, dtype=object)
                        unset = if_end == -1
                        apply_ac = unset & is_ac
                        if_end[apply_ac] = 1
                        strand[apply_ac] = '-'
                        unset = if_end == -1
                        apply_ct = unset & is_ct
                        if_end[apply_ct] = 0
                        strand[apply_ct] = '-'
                        unset = if_end == -1
                        apply_ag = unset & is_ag
                        if_end[apply_ag] = 1
                        strand[apply_ag] = '+'
                        unset = if_end == -1
                        apply_gt = unset & is_gt
                        if_end[apply_gt] = 0
                        strand[apply_gt] = '+'

                        target_ids = target_ids_all[batch_index]
                        start_pos = start_pos_all[batch_index]
                        if np.issubdtype(start_pos_all.dtype, np.number):
                            position = start_pos + pos_index
                        else:
                            position = np.empty(valid_indices.shape[0], dtype=object)
                            for idx in range(valid_indices.shape[0]):
                                start_val = _to_scalar(start_pos[idx])
                                try:
                                    position[idx] = int(start_val) + int(pos_index[idx])
                                except (TypeError, ValueError):
                                    position[idx] = start_val

                        scores = predictions_all[batch_index, pos_index].astype(float)
                        sample_indices = sample_indices_all[batch_index]
                        rows = list(
                            zip(
                                sample_indices,
                                pos_index,
                                target_ids,
                                position,
                                scores,
                                if_end,
                                strand,
                            )
                        )
                        if return_rows:
                            rows_out.extend(rows)
                        elif csv_writer:
                            for start in range(0, len(rows), csv_buffer_size):
                                csv_buffer.extend(
                                    [
                                        [r[2], r[3], r[4], r[5], r[6]]
                                        for r in rows[start : start + csv_buffer_size]
                                    ]
                                )
                                flush_csv_buffer()
                        else:
                            for row in rows:
                                print(
                                    f"{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\t{row[6]}"
                                )
                        rows_written += len(rows)
                    t_write += time.perf_counter() - t_write_start

                if stage != 'test':
                    t_metrics_start = time.perf_counter()
                    batch_start = 0
                    if merge_window > 0:
                        for batch_idx, batch_pos_indices in enumerate(
                            batch_pos_indices_list
                        ):
                            if batch_pos_indices[0].size == 0:
                                batch_start += batch_sizes[batch_idx]
                                continue
                            for batch_id in np.unique(batch_pos_indices[0]):
                                batch_indices = np.where(
                                    batch_pos_indices[0] == batch_id
                                )[0]
                                global_batch_id = batch_start + batch_id
                                if if_sortscore:
                                    sorted_batch = sorted(
                                        batch_indices,
                                        key=lambda i: predictions_all[global_batch_id][
                                            batch_pos_indices[1][i]
                                        ],
                                        reverse=True,
                                    )
                                else:
                                    sorted_batch = sorted(
                                        batch_indices,
                                        key=lambda i: pileup_embed_all[global_batch_id][
                                            batch_pos_indices[1][i] - 1
                                        ][1],
                                        reverse=True,
                                    )
                                accepted_positions = []
                                for i in sorted_batch:
                                    pos_i = batch_pos_indices[1][i]
                                    if all(
                                        abs(pos_i - batch_pos_indices[1][j])
                                        > merge_window
                                        for j in accepted_positions
                                    ):
                                        accepted_positions.append(i)
                                    else:
                                        predictions_all[global_batch_id][pos_i] = 0
                            batch_start += batch_sizes[batch_idx]

                    total_samples += len(batch_pos_indices_list)
                    actuals_all = np.concatenate(actuals_batches, axis=0)

                    # per-sample binary metrics
                    for i in range(actuals_all.shape[0]):
                        sample_actuals = actuals_all[i]
                        sample_predictions = predictions_all[i]
                        binary_predictions = (sample_predictions >= 0.5).astype(int)
                        if np.any(sample_actuals == 0) and np.any(sample_actuals == 1):
                            bw_accuracy = accuracy_score(
                                sample_actuals, binary_predictions
                            )
                            bw_f1 = f1_score(
                                sample_actuals, binary_predictions, average="macro"
                            )
                            bw_precision = precision_score(
                                sample_actuals,
                                binary_predictions,
                                average="macro",
                                zero_division=0,
                            )
                            bw_recall = recall_score(
                                sample_actuals,
                                binary_predictions,
                                average="macro",
                                zero_division=0,
                            )
                            bw_mcc = matthews_corrcoef(
                                sample_actuals, binary_predictions
                            )
                            tn, fp, fn, tp = confusion_matrix(
                                sample_actuals, binary_predictions
                            ).ravel()
                            total_tn += tn
                            total_tp += tp
                            total_fn += fn
                            total_fp += fp
                            bw_auc = roc_auc_score(sample_actuals, sample_predictions)
                            total_bw_accuracy += bw_accuracy
                            total_bw_f1 += bw_f1
                            total_bw_precision += bw_precision
                            total_bw_recall += bw_recall
                            total_bw_mcc += bw_mcc
                            total_bw_auc += bw_auc
                            total_bw_samples += 1
                            t_metrics += time.perf_counter() - t_metrics_start

        if stage == 'test':
            flush_csv_buffer()  # Flush remaining buffered rows.
            if csv_path_obj:
                print(
                    f"Saved {rows_written} predicted sites to {csv_path_obj.resolve()}"
                )
            t_total = time.perf_counter() - t_total_start
            print(
                "Timing (s) - total: {:.2f}, infer: {:.2f}, concat: {:.2f}, "
                "filter: {:.2f}, write: {:.2f}".format(
                    t_total, t_infer, t_concat, t_filter, t_write
                )
            )
            return {
                "rows": rows_out if return_rows else None,
                "csv_rows": rows_written,
                "csv_path": str(csv_path_obj.resolve()) if csv_path_obj else None,
            }

    finally:
        if csv_file is not None:
            csv_file.close()

    avg_bw_accuracy = total_bw_accuracy / total_bw_samples
    total_bw_precision = total_tp / (total_tp + total_fp + 1e-8)
    total_bw_recall = total_tp / (total_tp + total_fn + 1e-8)
    total_bw_f1 = (
        2
        * total_bw_precision
        * total_bw_recall
        / (total_bw_recall + total_bw_precision + 1e-8)
    )
    avg_bw_mcc = total_bw_mcc / total_bw_samples
    avg_bw_auc = total_bw_auc / total_bw_samples

    t_total = time.perf_counter() - t_total_start
    print(
        "Timing (s) - total: {:.2f}, infer: {:.2f}, concat: {:.2f}, "
        "filter: {:.2f}, metrics: {:.2f}".format(
            t_total, t_infer, t_concat, t_filter, t_metrics
        )
    )

    return {
        "Accuracy_bw": avg_bw_accuracy,
        "MCC_bw": avg_bw_mcc,
        "AUC_bw": avg_bw_auc,
        "F1_bw": total_bw_f1,
        "Precision_bw": total_bw_precision,
        "Recall_bw": total_bw_recall,
        "total_tn": total_tn,
        "total_tp": total_tp,
        "total_fn": total_fn,
        "total_fp": total_fp,
    }

# ----------------------------------------------------------------------
#  LightningModule
# ----------------------------------------------------------------------
class SpliceLightning(pl.LightningModule):
    def __init__(
        self,
        pretrained_ckpt_dir: str,
        pretrained_name: str = "hyenadna-medium-160k-seqlen",
        n_classes: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 100,
        backbone_cfg=None,
        model_path: str = "",
        stage: str = "test",
        #alpha: float =  min(0.25/max(2/10000,1e-6),0.99)
    ):
        super().__init__()
        self.save_hyperparameters()
        self.stage = stage

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device:", device)

        # 1) model ----------------------------------------------------
        self.model = HyenaDNAPreTrainedModel.from_pretrained(
            pretrained_ckpt_dir,
            pretrained_name,
            download=False,
            config=backbone_cfg,
            device=device,
            use_head=False,
            n_classes=n_classes,
        )
        self.model.stage = stage
        if model_path:
            checkpoint = torch.load(model_path, map_location="cpu")
            #self.model.load_state_dict({k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()})
            print("Loaded weights from", model_path)

        # freeze backbone if needed
        for p in self.model.backbone.parameters():
            p.requires_grad = False

        # 2) loss -----------------------------------------------------
        self.loss_fn = torchvision.ops.focal_loss.sigmoid_focal_loss

    # forward ---------------------------------------------------------
    def forward(self, x, pileup_embed=None):
        return self.model(x, pileup_embed=pileup_embed)

    # training --------------------------------------------------------
    def training_step(self, batch, batch_idx):
        seq, y, pileup = batch
        logits = self(seq, pileup_embed=pileup)
        loss = self.loss_fn(logits.squeeze(), y.squeeze(), alpha=0.8, reduction="mean")
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # validation / test ----------------------------------------------
    def validation_step(self, batch, batch_idx):
        if self.stage == "test":
            pass
        else:
            seq, y, pileup = batch
            logits = self(seq, pileup_embed=pileup)
            loss = self.loss_fn(logits.squeeze(), y.squeeze(), alpha=0.8, reduction="mean")
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return loss
     

    test_step = validation_step

    # optimizers ------------------------------------------------------
    def configure_optimizers(self):
        opt = optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sch}


# ----------------------------------------------------------------------
#  Callback that runs evaluate() every train / val / test epoch
# ----------------------------------------------------------------------
class _IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        return (idx, *sample)


class EvaluateCallback(pl.Callback):
    """Run the heavy evaluate() routine after each epoch and log the metrics.

    Metrics are prefixed with train_, val_, or test_ so they don’t clash
    with Lightning’s built-in logging.
    """

    def __init__(
        self,
        run_on_train: bool = False,
        run_on_val: bool = True,
        run_on_test: bool = True,
        test_csv_path: Optional[str] = "predictions.tsv",
    ):
        super().__init__()
        self.run_on_train = run_on_train
        self.run_on_val   = run_on_val
        self.run_on_test  = run_on_test
        self.test_csv_path = test_csv_path

    def _build_distributed_loader(self, loader, trainer):
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        dataset = _IndexedDataset(loader.dataset)
        sampler = DistributedSampler(
            dataset,
            num_replicas=trainer.world_size,
            rank=trainer.global_rank,
            shuffle=False,
            drop_last=False,
        )
        prefetch_factor = (
            getattr(loader, "prefetch_factor", None)
            if loader.num_workers > 0
            else None
        )
        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=getattr(loader, "pin_memory", False),
            drop_last=getattr(loader, "drop_last", False),
            persistent_workers=getattr(loader, "persistent_workers", False),
            prefetch_factor=prefetch_factor,
            collate_fn=getattr(loader, "collate_fn", None),
        )

    # ------------- internal helper ---------------------------------
    def _run_eval(self, trainer, pl_module, loader, prefix: str):
        device = pl_module.device
        is_distributed = trainer.world_size > 1
        if prefix == "test" and is_distributed:
            loader = self._build_distributed_loader(loader, trainer)
            results = evaluate(
                pl_module,
                loader,
                pl_module.loss_fn,
                device,
                prefix,
                csv_path=None,
                return_rows=True,
            )
            rows = results.get("rows") or []
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                gathered = [None for _ in range(trainer.world_size)]
                torch.distributed.all_gather_object(gathered, rows)
            else:
                gathered = [rows]

            if not trainer.is_global_zero:
                return

            all_rows = [row for part in gathered for row in part]
            all_rows.sort(key=lambda r: (r[0], r[1]))

            csv_path_obj = Path(self.test_csv_path)
            csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with csv_path_obj.open("w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file, delimiter="\t")
                csv_writer.writerows(
                    [[r[2], r[3], r[4], r[5], r[6]] for r in all_rows]
                )
            print(
                f"Saved {len(all_rows)} predicted sites to {csv_path_obj.resolve()}"
            )
            if trainer.logger:
                trainer.logger.log_metrics(
                    {f"{prefix}_csv_rows": len(all_rows)},
                    step=trainer.global_step,
                )
            return

        # run only on rank-0 so we don’t duplicate work in DDP
        if not trainer.is_global_zero:
            return
        csv_path = self.test_csv_path if prefix == "test" else None
        results = evaluate(
            pl_module,
            loader,
            pl_module.loss_fn,
            device,
            prefix,
            csv_path=csv_path,
        )
        if not results:
            return
        # add prefix and send to logger (W&B / TensorBoard …)
        if trainer.logger:
            numeric_results = {
                f"{prefix}_{k}": v
                for k, v in results.items()
                if isinstance(v, (int, float, np.floating))
            }
            if numeric_results:
                trainer.logger.log_metrics(numeric_results, step=trainer.global_step)

    # ------------- hooks --------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module):
        if self.run_on_train:
            loader = trainer.datamodule.train_dataloader()
            self._run_eval(trainer, pl_module, loader, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.run_on_val:
            loader = trainer.datamodule.val_dataloader()
            self._run_eval(trainer, pl_module, loader, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        if self.run_on_test:
            loader = trainer.datamodule.test_dataloader()
            self._run_eval(trainer, pl_module, loader, "test")

# ----------------------------------------------------------------------
#  LightningDataModule
# ----------------------------------------------------------------------
class SeqDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        stage: str = "train",
        center_len: int = 10000,
        batch_size: int = 28,
        num_workers: int = 8,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.data_dir, self.tokenizer, self.stage = data_dir, tokenizer, stage
        self.center_len, self.batch_size, self.num_workers = (
            center_len,
            batch_size,
            num_workers,
        )
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        if self.stage == 'test':
            from utils.dataset_inference import SeqTextDataset
        else:
            from utils.dataset_train import SeqTextDataset

        if stage in (None, "fit"):
            self.train_set = SeqTextDataset(
                f"{self.data_dir}/training/train",
                self.tokenizer,
                flanking=0,
                center=self.center_len,
            )
            self.val_set = SeqTextDataset(
                f"{self.data_dir}/training/val",
                self.tokenizer,
                flanking=0,
                center=self.center_len,
            )
        if stage in (None, "test"):
            self.test_set = SeqTextDataset(
                f"{self.data_dir}/training/test",
                self.tokenizer,
                flanking=0,
                center=self.center_len,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# ----------------------------------------------------------------------
#  Main – replaces run_train()
# ----------------------------------------------------------------------
def cli_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="", help="optional ckpt to fine-tune from")
    parser.add_argument("--stage", default="test", help="train/test")
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--accumulate_grad_batches", default=2)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--prefetch_factor", default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="path to the directory with extracted features data",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, ValueError):
            pass

    # ------------- tokenizer -----------------
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T"],
        model_max_length=10000 + 2,
        add_special_tokens=False,
        padding_side="left",
    )

    # ------------- data ----------------------
    data_dir = args.data_dir
    dm = SeqDataModule(
        data_dir,
        tokenizer,
        stage=args.stage,
        batch_size=args.batch_size,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
    )

    # ------------- model ---------------------
    lit_model = SpliceLightning(
        pretrained_ckpt_dir="./checkpoints",
        model_path=args.model_path,
        stage=args.stage,
        lr=1e-3,
        weight_decay=0.01,
        max_epochs=100,
    )
    if args.compile and hasattr(torch, "compile"):
        lit_model.model = torch.compile(lit_model.model)

    os.environ["WANDB_DATA_DIR"] = ''

    # ------------- logger --------------------
    wandb_logger = False

    early_stop = EarlyStopping(
	monitor="val_loss",  
	mode="min",         
	patience=15,  
	verbose=True,
    )
    trainer_kwargs = dict(
        max_epochs=100,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        devices="auto",
        accelerator="auto",
        strategy="ddp_find_unused_parameters_false"
        if torch.cuda.device_count() > 1
        else "auto",
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=[
            ModelCheckpoint(
                save_top_k=-1,
                monitor="val_loss",
                mode="min",
            ),
            early_stop,
            EvaluateCallback(test_csv_path=f"{data_dir}.predictions.tsv"),
        ],
    )

    # ------------- fit / test ----------------
    if args.model_path=='':
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(lit_model, datamodule=dm)
    elif args.model_path and args.stage=='test':
        trainer = pl.Trainer(**trainer_kwargs, limit_test_batches=0.01)
        trainer.test(lit_model, datamodule=dm, ckpt_path=args.model_path)
    else:
        #  continue a ckpt
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(lit_model, datamodule=dm, ckpt_path=args.model_path)


if __name__ == "__main__":
    cli_main()

