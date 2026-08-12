import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import types

from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaConfig
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .attention_utils import AttrDict

class LlamaAttention(nn.Module):
    """
    Multi-headed attention with per-layer flexible QK / VO dims and RoPE.
    ---------------------------
    num_heads       Q == K head count (and output head count)
    head_dim_qk     per-head channel depth used by Q and K
    head_dim_vo     per-head channel depth used by V and O
    qk_out_dim      num_heads * head_dim_qk
    vo_out_dim      num_heads * head_dim_vo
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config    = config
        self.layer_idx = layer_idx

        if isinstance(config, dict):
            self.hidden_size             = config["hidden_size"]
            self.num_heads               = config["heads"]
            self.head_dim_qk             = config.get("q_dim", self.hidden_size // self.num_heads)
            self.head_dim_vo             = config.get("v_dim", self.hidden_size // self.num_heads)
            self.attention_bias          = config.get("attention_bias", False)
            self.attention_dropout       = config.get("attention_dropout", 0.0)
            self.max_position_embeddings = config.get("max_position_embeddings", 4096)
            self.rope_theta              = config.get("rope_theta")
            self.rope_scaling            = config.get("rope_scaling")
            self.partial_rotary_factor   = config.get("partial_rotary_factor")
            self.rope_parameters         = config.get("rope_parameters")
            scale_dim_override           = config.get("scale_dim", None)
            qk_dims_cfg                  = config.get("qk_dims", None)
            vo_dims_cfg                  = config.get("vo_dims", None)
            pad_dim_cfg                  = config.get("attn_pad_dim", None)
            qk_scale_cfg                 = config.get("qk_scale", None)
        else:
            self.hidden_size             = config.hidden_size
            self.num_heads               = config.heads
            self.head_dim_qk             = getattr(config, "q_dim", self.hidden_size // self.num_heads)
            self.head_dim_vo             = getattr(config, "v_dim", self.hidden_size // self.num_heads)
            self.attention_bias          = getattr(config, "attention_bias", False)
            self.attention_dropout       = getattr(config, "attention_dropout", 0.0)
            self.max_position_embeddings = getattr(config, "max_position_embeddings", 4096)
            self.rope_theta              = getattr(config, "rope_theta", None)
            self.rope_scaling            = getattr(config, "rope_scaling", None)
            self.partial_rotary_factor   = getattr(config, "partial_rotary_factor", None)
            self.rope_parameters         = getattr(config, "rope_parameters", None)
            scale_dim_override           = getattr(config, "scale_dim", None)
            qk_dims_cfg                  = getattr(config, "qk_dims", None)
            vo_dims_cfg                  = getattr(config, "vo_dims", None)
            pad_dim_cfg                  = getattr(config, "attn_pad_dim", None)
            qk_scale_cfg                 = getattr(config, "qk_scale", None)

        self.is_causal = True
        # scale_dim override preserves training-time softmax scale when the
        # layer was sliced with zero-padding (see analogous note in
        # custom_attentions/dinov2_attention.py).
        scale_dim = scale_dim_override if scale_dim_override is not None else self.head_dim_qk
        self.scaling   = scale_dim ** -0.5
        # When True, the forward bypasses the 4-D attention_mask HF passes
        # in and uses (attn_mask=None, is_causal=True) so SDPA dispatches
        # to flash/efficient instead of the math kernel. Safe only for
        # prefill on un-padded inputs (q_len == kv_len, no padding mask),
        # which is the case for the benchmark and for typical eval runs.
        self.force_flash_attn = False

        # Ragged mode: per-head channel counts differ. Everything below sizes
        # off qk_out_dim / vo_out_dim, which become the SUM of the per-head
        # widths rather than num_heads * a single width, so the projections
        # stay compact. _set_ragged_dims derives the offsets and the
        # scatter/gather indices; when qk_dims is absent this is a no-op and
        # the module behaves exactly as before.
        self.ragged = qk_dims_cfg is not None
        if self.ragged:
            self._set_ragged_dims(list(qk_dims_cfg), list(vo_dims_cfg),
                                  pad_dim_cfg, qk_scale_cfg)
        else:
            self.qk_out_dim = self.num_heads * self.head_dim_qk
            self.vo_out_dim = self.num_heads * self.head_dim_vo

        self.q_proj = nn.Linear(self.hidden_size, self.qk_out_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.qk_out_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.vo_out_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.vo_out_dim,  self.hidden_size, bias=self.attention_bias)

        self.register_buffer("rope_cos", None)
        self.register_buffer("rope_sin", None)
        self._precompute_rope_tables()
        if torch.version.hip is not None:
            self.attn_dtype = torch.float16   # safer on MI250
        else:
            self.attn_dtype = torch.bfloat16
        assert self.rope_cos.dtype == torch.float32
        assert torch.isfinite(self.rope_cos).all()

    # ── Ragged (per-head channel counts) ─────────────────────────────────────

    def _set_ragged_dims(self, qk_dims, vo_dims, attn_pad_dim=None,
                         qk_scale=None):
        """Derive the compact-layout bookkeeping for per-head channel counts.

        Buffers are deleted and re-registered rather than copied into, because
        reconstruct_weights changes num_heads and the shapes move with it --
        the same reason Llama3Attention._set_group_sizes does it this way.

        Layout. The projections are COMPACT: q_proj emits sum(qk_dims)
        columns, laid out head after head. The attention core needs a
        rectangle, so the compact tensor is scattered into [B, H, D_pad] with
        each head occupying its own D_pad-wide slot:

            head h slot:  [ firsts | zeros | seconds | zeros ]
                          ^0       ^K/2    ^D_pad/2  ^D_pad/2+K/2

        The two RoPE halves must land at 0 and D_pad/2 because _apply_rope
        splits a head at its midpoint. qk_pad_index encodes exactly that
        permutation, so the scatter is one index_copy_ rather than a loop.
        """
        H = len(qk_dims)
        if len(vo_dims) != H:
            raise ValueError(f"qk_dims/vo_dims length mismatch: {H} vs {len(vo_dims)}")
        if H != self.num_heads:
            raise ValueError(f"ragged dims cover {H} heads, module has {self.num_heads}")
        if any(d <= 0 for d in qk_dims) or any(d <= 0 for d in vo_dims):
            raise ValueError("every kept head needs at least one qk and one vo channel")
        if any(d % 2 for d in qk_dims):
            raise ValueError(f"qk widths must be even (RoPE pairs), got {qk_dims}")

        D_pad = attn_pad_dim or max(max(qk_dims), max(vo_dims))
        D_pad += (-D_pad) % 2
        if D_pad < max(max(qk_dims), max(vo_dims)):
            raise ValueError(f"attn_pad_dim {D_pad} smaller than the widest head")

        self.qk_out_dim = int(sum(qk_dims))
        self.vo_out_dim = int(sum(vo_dims))
        self.attn_pad_dim = int(D_pad)
        # Reporting values. The layout is per-head from here on; these exist so
        # every current reader (analysis scripts, config writers) keeps working.
        self.head_dim_qk = int(max(qk_dims))
        self.head_dim_vo = int(max(vo_dims))

        half = D_pad // 2
        qk_idx, vo_idx = [], []
        for h, (kq, kv) in enumerate(zip(qk_dims, vo_dims)):
            b = h * D_pad
            p = kq // 2
            qk_idx += [b + j for j in range(p)]              # first halves
            qk_idx += [b + half + j for j in range(p)]       # second halves
            vo_idx += [b + j for j in range(kv)]

        # Per-head softmax temperature. SDPA takes a scalar `scale`, so the
        # per-head factor is applied to q before the call and `sdpa_scale` is
        # 1.0. Storing it explicitly (rather than deriving 1/sqrt(width)) is
        # what lets a zero-padded slice keep the temperature of the LIVE
        # channel count while its stored width is larger.
        if qk_scale is None:
            qk_scale = [float(d) ** -0.5 for d in qk_dims]
        # Held in fp64 and cast at use. Storing it at fp32 would round the
        # softmax temperature to ~1e-7 relative, which is invisible in bf16 but
        # puts a floor under any fp64 equivalence test -- and an equivalence
        # test that cannot go below 1e-7 cannot distinguish a real semantic
        # error from rounding.
        scale_flat = torch.cat([
            torch.full((d,), float(s), dtype=torch.float64)
            for d, s in zip(qk_dims, qk_scale)])

        for name, val, persist in (
            ("qk_dims", torch.tensor(qk_dims, dtype=torch.long), False),
            ("vo_dims", torch.tensor(vo_dims, dtype=torch.long), False),
            ("qk_pad_index", torch.tensor(qk_idx, dtype=torch.long), False),
            ("vo_pad_index", torch.tensor(vo_idx, dtype=torch.long), False),
            ("qk_scale_flat", scale_flat, False),
        ):
            if name in self._buffers:
                del self._buffers[name]
            self.register_buffer(name, val, persistent=persist)

        self.qk_scale_list = [float(s) for s in qk_scale]
        self.ragged = True
        self.sdpa_scale = 1.0

    def _scatter_heads(self, x, index):
        """Compact [B, T, S] -> rectangular [B, H, T, D_pad]."""
        B, T, _ = x.shape
        out = x.new_zeros(B, T, self.num_heads * self.attn_pad_dim)
        out.index_copy_(2, index, x)
        return out.view(B, T, self.num_heads, self.attn_pad_dim).transpose(1, 2)

    # ── RoPE ─────────────────────────────────────────────────────────────────

    def _precompute_rope_tables(self):
        """Build the cos/sin tables from per-head frequencies.

        Everything routes through `rope_inv_freq`, a [H, W/2] buffer holding
        each head's rotation frequencies. In the dense case every row is the
        standard inv_freq and the result is identical to broadcasting one
        table across heads. In the ragged case each head kept a different
        SUBSET of the original pairs, and a pair's frequency depends on its
        ORIGINAL index -- so the surviving frequencies have to be carried
        per head, not recomputed from the new (smaller) width. Padding slots
        hold frequency 0, i.e. cos=1/sin=0, an identity rotation on the zero
        channels that sit there.

        Keeping this the single construction path means a ragged checkpoint
        whose rope buffers get rebuilt (create_llama_and_tokenizer does this
        when they are missing or non-finite) is rebuilt CORRECTLY rather than
        silently regenerating dense tables over a ragged layout.
        """
        base   = float(self.rope_theta) if self.rope_theta else 10_000.0
        device = torch.device("cpu")
        dtype  = torch.float32

        # Table width: D_pad/2 when ragged (each head occupies a D_pad-wide
        # slot in the attention rectangle), head_dim_qk/2 otherwise.
        width = (self.attn_pad_dim if getattr(self, "ragged", False)
                 else self.head_dim_qk) // 2

        inv_freq = getattr(self, "rope_inv_freq", None)
        # BOTH dims, not just the width. Checking only the width let the 'head'
        # arm through: it prunes heads while keeping 128/128 channels, so the
        # width was unchanged and a stale [H_orig, W] buffer looked valid, got
        # saved, and failed on reload against a config with fewer heads
        # (job 51102680: [32, 64] vs [20, 64]).
        if inv_freq is not None and tuple(inv_freq.shape) != (self.num_heads, width):
            inv_freq = None                       # geometry moved; rebuild
        if inv_freq is None:
            if getattr(self, "ragged", False):
                # A ragged layer's frequencies are per head and depend on which
                # ORIGINAL pairs that head kept, which is not derivable from
                # the new width. They arrive with the checkpoint (the slicer
                # emits them); allocate zeros so the buffer SHAPE matches and
                # load_state_dict can fill them. Zero frequency is cos=1/sin=0,
                # an identity rotation -- wrong but inert, never NaN.
                inv_freq = torch.zeros(self.num_heads, width,
                                       device=device, dtype=dtype)
            else:
                dim = self.head_dim_qk
                base_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                                         dtype=dtype) / dim))
                inv_freq = base_freq[None].repeat(self.num_heads, 1)   # (H, dim/2)
            if "rope_inv_freq" in self._buffers:
                del self._buffers["rope_inv_freq"]
            self.register_buffer("rope_inv_freq", inv_freq, persistent=True)
        inv_freq = inv_freq.to(device=device, dtype=dtype)

        pos    = torch.arange(self.max_position_embeddings, device=device, dtype=dtype)
        angles = pos[None, :, None] * inv_freq[:, None, :]          # (H, P, W/2)
        cos, sin = torch.cos(angles), torch.sin(angles)

        for buf in ("rope_cos", "rope_sin"):
            if buf in self._buffers:
                del self._buffers[buf]
        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)

    def _apply_rope(self, x: torch.Tensor, position_ids=None, return_cos_sin: bool = False):
        """Apply rotary position embeddings. x: (B, H, L, head_dim_qk).

        cos/sin tables are stored in fp32 for precision; the actual
        rotation runs in x's dtype (typically bf16), which keeps the
        bf16 matmul path on A100 instead of forcing the per-token mul
        through the fp32 pipeline.
        """
        B, nh, L, hd = x.shape
        device = x.device
        P = hd // 2

        assert nh == self.rope_cos.shape[0], \
            f"Head count mismatch: x has {nh} heads, rope has {self.rope_cos.shape[0]}"

        pos_idx = (
            torch.arange(L, device=device)
            if position_ids is None
            else (position_ids[0] if position_ids.dim() == 2 else position_ids).to(device)
        )
        pos_idx = pos_idx.clamp(0, self.max_position_embeddings - 1)

        with torch.autocast(device_type=device.type, enabled=False):
            cos_pairs = self.rope_cos[:, pos_idx, :]   # (H, L, P) fp32
            sin_pairs = self.rope_sin[:, pos_idx, :]

        # Cast tables to x's dtype so the rotation math runs in bf16.
        cos_b = cos_pairs.to(x.dtype).unsqueeze(0).expand(B, -1, -1, -1)
        sin_b = sin_pairs.to(x.dtype).unsqueeze(0).expand(B, -1, -1, -1)
        x1, x2 = x[..., :P], x[..., P:]
        x_rot = torch.cat([x1 * cos_b - x2 * sin_b,
                            x2 * cos_b + x1 * sin_b], dim=-1)

        return (x_rot, cos_pairs, sin_pairs) if return_cos_sin else x_rot


    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len = hidden_states.shape[:-1]
        input_dtype = hidden_states.dtype

        # 1. Project Q, K, V
        if self.ragged:
            # Compact projections: [B, T, sum(K_h)]. The per-head softmax
            # temperature goes on q HERE, on the smallest tensor available and
            # before the scatter -- rotation is linear, so s*R(q) == R(s*q)
            # and the ordering is free. SDPA's `scale` is a scalar, so this is
            # the only place a [H] vector can enter one fused call.
            qc = self.q_proj(hidden_states) * self.qk_scale_flat.to(hidden_states.dtype)
            q = self._scatter_heads(qc, self.qk_pad_index)
            k = self._scatter_heads(self.k_proj(hidden_states), self.qk_pad_index)
            v = self._scatter_heads(self.v_proj(hidden_states), self.vo_pad_index)
        else:
            q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim_qk).transpose(1, 2)
            k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim_qk).transpose(1, 2)
            v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim_vo).transpose(1, 2)

        # 2. Apply RoPE in the projection dtype (bf16) — _apply_rope casts
        # cos/sin to x.dtype so the rotation runs on tensor cores, not
        # the slower fp32 path.
        q, cos_pairs, sin_pairs = self._apply_rope(q, position_ids, return_cos_sin=True)
        k = self._apply_rope(k, position_ids, return_cos_sin=False)

        # 3. KV Cache update (during inference)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin_pairs, "cos": cos_pairs, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        # 4. Scaled Dot-Product Attention
        # PyTorch SDPA will throw an error if both `is_causal=True` and `attn_mask` are provided.
        # HuggingFace passes a 4D mask that already includes causality, so by default we
        # forward it and turn off is_causal — but this routes SDPA to the math kernel.
        # When force_flash_attn is set (benchmark / un-padded eval), we drop the mask
        # and let SDPA pick flash/efficient via is_causal=True.
        if self.force_flash_attn and q_len > 1:
            sdpa_mask = None
            is_causal = True
        else:
            is_causal = self.is_causal if attention_mask is None and q_len > 1 else False
            sdpa_mask = attention_mask

        # Flash/efficient SDPA needs q.size(-1) == k.size(-1) == v.size(-1), and
        # head_dim ∈ {8,16,32,64,96,128,…}. Structurally-pruned models have
        # irregular per-layer q_dim (e.g. 22, 30, 38, …) and often q_dim != v_dim,
        # so SDPA falls to the math kernel and runs ~5–10× slower. When
        # force_flash_attn is set, zero-pad Q/K/V to a uniform target (default 128)
        # so flash engages, then slice the V dim back. Math is preserved because
        # the padded channels carry zeros, contributing nothing to the dot product
        # or the value mixture. We keep `scale = head_dim_qk ** -0.5` (the original
        # softmax temperature) so the score distribution is unchanged.
        flash_pad_target = 128
        pad_qk = pad_vo = 0
        if self.ragged:
            # The scatter already produced a rectangle of width attn_pad_dim,
            # with q == k == v width and zeros in every unused slot -- exactly
            # the shape this block would otherwise construct, so it is skipped
            # rather than applied twice. Widen further only if attn_pad_dim is
            # not a size flash accepts.
            if self.force_flash_attn and q_len > 1 and self.attn_pad_dim < flash_pad_target:
                extra = flash_pad_target - self.attn_pad_dim
                q = nn.functional.pad(q, (0, extra))
                k = nn.functional.pad(k, (0, extra))
                v = nn.functional.pad(v, (0, extra))
                pad_vo = extra
        elif self.force_flash_attn and q_len > 1:
            pad_qk = flash_pad_target - self.head_dim_qk
            pad_vo = flash_pad_target - self.head_dim_vo
            if pad_qk > 0:
                q = nn.functional.pad(q, (0, pad_qk))
                k = nn.functional.pad(k, (0, pad_qk))
            if pad_vo > 0:
                v = nn.functional.pad(v, (0, pad_vo))

        attn_output = nn.functional.scaled_dot_product_attention(
            q.to(self.attn_dtype),
            k.to(self.attn_dtype),
            v.to(self.attn_dtype),
            attn_mask=sdpa_mask,
            is_causal=is_causal,
            # Ragged already folded the per-head 1/sqrt(K_h) into q, so the
            # kernel-level scale must be 1.0 or it would be applied twice.
            scale=(self.sdpa_scale if self.ragged else self.scaling),
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        if self.ragged:
            # Rectangle -> compact [B, T, sum(K_vo)]. index_select inverts the
            # v scatter; the slots it skips are exactly the zero ones.
            attn_output = (attn_output.transpose(1, 2)
                           .reshape(bsz, q_len, -1)
                           .index_select(2, self.vo_pad_index))
            return self.o_proj(attn_output.to(input_dtype)), None

        if pad_vo > 0:
            attn_output = attn_output[..., :self.head_dim_vo]

        expected = (bsz, self.num_heads, q_len, self.head_dim_vo)
        if attn_output.size() != expected:
            raise ValueError(f"`attn_output` should be {expected}, got {attn_output.size()}")

        # 5. Output Projection
        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(bsz, q_len, self.vo_out_dim)
            .to(input_dtype)
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, None   # (output, attn_weights)
