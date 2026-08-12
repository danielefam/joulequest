"""
Generic attention implementations:
  - RaggedFlashAttentionFunction
  - PythonFlashAttention
  - FlexibleFlashAttention
  - MultiheadAttention  (drop-in replacement for nn.MultiheadAttention)
  - Attention           (lightweight ViT-style block)
  - Block               (ViT-style transformer block wrapping Attention)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import dropout
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# MultiheadAttention  (flexible drop-in for nn.MultiheadAttention)
# ---------------------------------------------------------------------------

class MultiheadAttention(Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        head_dim_qk=None,
        head_dim_vo=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be > 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads}"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads  = num_heads

        self.dropout = dropout
        self.batch_first = batch_first
        self.add_zero_attn = add_zero_attn

        self.head_dim_qk = head_dim_qk if head_dim_qk is not None else embed_dim // self.num_heads
        self.head_dim_vo = head_dim_vo if head_dim_vo is not None else embed_dim // self.num_heads

        self.qk_out_dim = self.num_heads * self.head_dim_qk
        self.vo_out_dim = self.num_heads * self.head_dim_vo

        self.in_proj_weight = nn.Linear(
            self.embed_dim,
            self.qk_out_dim*2 + self.vo_out_dim,
            bias=bias,
            **factory_kwargs,
        )
        self.out_proj = nn.Linear(self.vo_out_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        if key is query and value is query:
            proj = self.in_proj_weight(query)
            q = proj[..., :self.qk_out_dim]
            k = proj[..., self.qk_out_dim:2*self.qk_out_dim]
            v = proj[..., 2*self.qk_out_dim:]
        else:
            w, b = self.in_proj_weight.weight, self.in_proj_weight.bias
            q = F.linear(query, w[:self.qk_out_dim],
                         b[:self.qk_out_dim] if b is not None else None)
            k = F.linear(key,   w[self.qk_out_dim:2*self.qk_out_dim],
                         b[self.qk_out_dim:2*self.qk_out_dim] if b is not None else None)
            v = F.linear(value, w[2*self.qk_out_dim:],
                         b[2*self.qk_out_dim:] if b is not None else None)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim_qk).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim_qk).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim_vo).transpose(0, 1)

        if self.add_zero_attn:
            zero_shape = (bsz * self.num_heads, 1, self.head_dim_qk)
            k = torch.cat([k, torch.zeros(zero_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_shape, dtype=v.dtype, device=v.device)], dim=1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len)
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )
            attn_mask = key_padding_mask if attn_mask is None else attn_mask + key_padding_mask

        dropout_p = self.dropout if self.training else 0.0

        if need_weights:
            B, Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))
            assert not (is_causal and attn_mask is None), \
                "is_causal not implemented for need_weights"
            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if dropout_p > 0.0:
                attn_output_weights = dropout(attn_output_weights, p=dropout_p)
            attn_output = torch.bmm(attn_output_weights, v)
            attn_output = (
                attn_output.transpose(0, 1)
                .contiguous()
                .view(tgt_len * bsz, self.num_heads * self.head_dim_vo)
            )
            attn_output = self.out_proj(attn_output).view(tgt_len, bsz, -1)
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            if not is_batched:
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            if self.batch_first and is_batched:
                return attn_output.transpose(1, 0), attn_output_weights
            return attn_output, attn_output_weights

        else:
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

            q = q.view(bsz, self.num_heads, tgt_len, self.head_dim_qk)
            k = k.view(bsz, self.num_heads, src_len, self.head_dim_qk)
            v = v.view(bsz, self.num_heads, src_len, self.head_dim_vo)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, is_causal,
                scale=1.0 / math.sqrt(self.head_dim_qk),
            )
            attn_output = (
                attn_output.permute(2, 0, 1, 3)
                .contiguous()
                .view(bsz * tgt_len, self.num_heads * self.head_dim_vo)
            )
            attn_output = self.out_proj(attn_output).view(tgt_len, bsz, -1)
            if not is_batched:
                attn_output = attn_output.squeeze(1)
            if self.batch_first and is_batched:
                return attn_output.transpose(1, 0), None
            return attn_output, None
    
    def reconstruct_weights(self, qkv_weights, qkv_bias, o_weights, embed_dim, num_heads, head_dims, device, **kwargs):
        #shapes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim_qk = head_dims[0]
        self.head_dim_vo = head_dims[1]
        self.qk_out_dim = self.head_dim_qk *self.num_heads
        self.vo_out_dim = self.head_dim_vo *self.num_heads

        #qkv projection
        self.in_proj_weight = nn.Linear(self.embed_dim, 2*self.qk_out_dim + self.vo_out_dim, bias=(qkv_bias is not None), device=device)
        self.in_proj_weight.weight.data.copy_(qkv_weights)
        if qkv_bias is not None:
                self.in_proj_weight.bias.data.copy_(qkv_bias)
        
        #o projection
        old_o_bias = self.out_proj.bias.data.clone() if self.out_proj.bias is not None else None
        self.out_proj = nn.Linear(self.vo_out_dim, self.embed_dim, bias=(old_o_bias is not None), device=device)
        self.out_proj.weight.data.copy_(o_weights)
        if old_o_bias is not None:
            self.out_proj.bias.data.copy_(old_o_bias)