import torch
from torch import nn


class SelfAttention(nn.Module):
    """Unary self-attention layer for the common query/key/value input."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, inputs):
        return self.attention(
            inputs,
            inputs,
            inputs,
            need_weights=False,
        )[0]


class RotarySelfAttention(nn.Module):
    """Self-attention with rotary positional embeddings on queries and keys."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Rotary attention head dimension must be even")
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def _apply_rotary_embedding(self, inputs):
        sequence_length = inputs.shape[0]
        positions = torch.arange(
            sequence_length,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        frequencies = torch.arange(
            0,
            self.head_dim,
            2,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        inverse_frequencies = 1.0 / (10000 ** (frequencies / self.head_dim))
        angles = positions[:, None] * inverse_frequencies[None, :]
        cosines = angles.cos()[:, None, None, :]
        sines = angles.sin()[:, None, None, :]
        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]
        rotated = torch.stack(
            (even * cosines - odd * sines, even * sines + odd * cosines),
            dim=-1,
        )
        return rotated.flatten(start_dim=-2)

    def forward(self, inputs):
        sequence_length, batch_size, _ = inputs.shape
        query = self.query(inputs).reshape(
            sequence_length, batch_size, self.num_heads, self.head_dim
        )
        key = self.key(inputs).reshape(
            sequence_length, batch_size, self.num_heads, self.head_dim
        )
        value = self.value(inputs).reshape(
            sequence_length, batch_size, self.num_heads, self.head_dim
        )
        query = self._apply_rotary_embedding(query)
        key = self._apply_rotary_embedding(key)
        attention_scores = torch.einsum("sbhd,tbhd->bhst", query, key)
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = torch.einsum("bhst,tbhd->sbhd", attention_weights, value)
        return self.output(attended.reshape(sequence_length, batch_size, -1))