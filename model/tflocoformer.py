# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: Apache-2.0


import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")

class TFLocoformer_model(nn.Module):
    def __init__(
        self,
        num_spks,
        emb_dim,
        n_layers,
        ffn_type,
        ffn_hidden_dim,
    ):
        super().__init__()
        self.num_spks = num_spks

        self.encoder = STFTEncoder(
            n_fft=128,
            hop_length=64,
        )
        self.separator = TFLocoformerSeparator(
            num_spk=num_spks,
            n_layers=n_layers,
            emb_dim=emb_dim,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
        )
        self.decoder = ISTFTDecoder(
            n_fft=128,
            hop_length=64,
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        spec = self.encoder(x)  # [B, T, F]
        separated_specs = self.separator(spec)
        separated_waves = [self.decoder(spec) for spec in separated_specs]
        return separated_waves

class TFLocoformerSeparator(nn.Module):

    def __init__(
        self,
        num_spk: int = 2,
        n_layers: int = 6,
        # general setup
        emb_dim: int = 128,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,  # used only in RMSGroupNorm
        tf_order: str = "ft",
        # self-attention related
        n_heads: int = 4,
        flash_attention: bool = False,  # available when using mixed precision
        attention_dim: int = 128,
        # ffn related
        ffn_type: Union[str, list] = "swiglu_conv1d",
        ffn_hidden_dim: Union[int, list] = 384,
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        # others
        eps: float = 1.0e-5,
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"

        self._num_spk = num_spk
        self.n_layers = n_layers

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),  # gLN
        )

        assert attention_dim % n_heads == 0, (attention_dim, n_heads)
        rope_freq = RotaryEmbedding(attention_dim // n_heads)
        rope_time = RotaryEmbedding(attention_dim // n_heads)
        
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                TFLocoformerBlock(
                    rope_freq,
                    rope_time,
                    # general setup
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    tf_order=tf_order,
                    # self-attention related
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    # ffn related
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, num_spk * 2, ks, padding=padding)

    def forward(
        self,
        input: torch.Tensor,
    ) -> List[torch.Tensor]:
        if input.ndim == 3:
            # in case the input does not have channel dimension
            batch0 = input.unsqueeze(1)
        elif input.ndim == 4:
            assert input.shape[1] == 1, "Only monaural input is supported."
            batch0 = input.transpose(1, 2)  # [B, M, T, F]

        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.conv(batch)  # [B, -1, T, F]

        # separation
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.deconv(batch)  # [B, num_spk*2, T, F]
        batch = batch.view([n_batch, self.num_spk, 2, n_frames, n_freqs])

        batch = torch.complex(batch[:, :, 0], batch[:, :, 1])
        batch = [batch[:, src] for src in range(self.num_spk)]

        return batch

class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        rope_freq,
        rope_time,
        # general setup
        emb_dim=128,
        norm_type="rmsgroupnorm",
        num_groups=4,
        tf_order="ft",
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

        self.freq_path = LocoformerBlock(
            rope_freq,
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )
        self.frame_path = LocoformerBlock(
            rope_time,
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )

    def forward(self, input):

        if self.tf_order == "ft":
            output = self.freq_frame_process(input)
        else:
            output = self.frame_freq_process(input)

        return output

    def freq_frame_process(self, input):
        output = input.movedim(1, -1)  # (B, T, Q_old, H)
        output = self.freq_path(output)

        output = output.transpose(1, 2)  # (B, F, T, H)
        output = self.frame_path(output)
        return output.transpose(-1, 1)

    def frame_freq_process(self, input):
        # Input tensor, (n_batch, hidden, n_frame, n_freq)
        output = input.transpose(1, -1)  # (B, F, T, H)
        output = self.frame_path(output)

        output = output.transpose(1, 2)  # (B, T, F, H)
        output = self.freq_path(output)
        return output.movedim(-1, 1)


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        rope,
        # general setup
        emb_dim=128,
        norm_type="rmsgroupnorm",
        num_groups=4,
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        FFN = {
            "conv1d": ConvDeconv1d,
            "swiglu_conv1d": SwiGLUConvDeconv1d,
        }
        Norm = {
            "layernorm": nn.LayerNorm,
            "rmsgroupnorm": RMSGroupNorm,
        }
        assert norm_type in Norm, norm_type

        self.macaron_style = isinstance(ffn_type, list) and len(ffn_type) == 2
        if self.macaron_style:
            assert (
                isinstance(ffn_hidden_dim, list) and len(ffn_hidden_dim) == 2
            ), "Two FFNs required when using Macaron-style model"

        # initialize FFN
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])
        for f_type, f_dim in zip(ffn_type[::-1], ffn_hidden_dim[::-1]):
            assert f_type in FFN, f_type
            if norm_type == "rmsgroupnorm":
                self.ffn_norm.append(Norm[norm_type](num_groups, emb_dim, eps=eps))
            else:
                self.ffn_norm.append(Norm[norm_type](emb_dim, eps=eps))
            self.ffn.append(
                FFN[f_type](
                    emb_dim,
                    f_dim,
                    conv1d_kernel,
                    conv1d_shift,
                    dropout=dropout,
                )
            )

        # initialize self-attention
        if norm_type == "rmsgroupnorm":
            self.attn_norm = Norm[norm_type](num_groups, emb_dim, eps=eps)
        else:
            self.attn_norm = Norm[norm_type](emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    def forward(self, x):
        B, T, F, C = x.shape

        if self.macaron_style:
            input_ = x
            output = self.ffn_norm[-1](x)  # [B, T, F, C]
            output = self.ffn[-1](output)  # [B, T, F, C]
            output = output + input_
        else:
            output = x

        # Self-attention
        input_ = output
        output = self.attn_norm(output)
        output = output.view([B * T, F, C])
        output = self.attn(output)
        output = output.view([B, T, F, C]) + input_

        # FFN after self-attention
        input_ = output
        output = self.ffn_norm[0](output)  # [B, T, F, C]
        output = self.ffn[0](output)  # [B, T, F, C]
        output = output + input_

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # rotary positional encoding
        query, key = self.apply_rope(query, key)

        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key


class ConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.diff_ks = conv1d_kernel - conv1d_shift

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, conv1d_kernel, stride=conv1d_shift),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, s1, s2, h = x.shape
        x = x.view(b * s1, s2, h)
        x = x.transpose(-1, -2)
        x = self.net(x).transpose(-1, -2)
        x = x[..., self.diff_ks // 2 : self.diff_ks // 2 + s2, :]
        return x.view(b, s1, s2, h)


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        # cut necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output
    
class STFTEncoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=None, window="hann"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length else n_fft
        self.window = getattr(torch, f"{window}_window")(self.win_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            return_complex=True,
        )  # [B, F, T]
        return spec.transpose(1, 2)  # [B, T, F]

class ISTFTDecoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=None, window="hann"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length else n_fft
        self.window = getattr(torch, f"{window}_window")(self.win_length)

    def forward(self, spec: torch.Tensor, length: int = None) -> torch.Tensor:
        # spec: [B, T, F] (complex)
        spec = spec.transpose(1, 2)  # [B, F, T]
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(spec.device),
            length=length,
        )
        return audio