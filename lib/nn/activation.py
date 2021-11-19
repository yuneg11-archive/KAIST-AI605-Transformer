import math

import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "MultiheadAttention",
]


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.head_dim_scale = math.sqrt(self.head_dim)
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # Projection
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Reset parameters
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape

        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        query = F.linear(query.transpose(0, 1), w_q, b_q) \
                 .reshape(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        key   = F.linear(key.transpose(0, 1), w_k, b_k) \
                 .reshape(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        value = F.linear(value.transpose(0, 1), w_v, b_v) \
                 .reshape(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        if key_padding_mask is not None:
            key_padding_mask = \
                key_padding_mask.view(batch_size, 1, 1, src_len) \
                                .expand(-1, self.num_heads, -1, -1) \
                                .reshape(batch_size * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                if attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask.logical_or(key_padding_mask)
                else:
                    attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=torch.float) \
                             .masked_fill_(attn_mask, float("-inf"))

        attn = torch.bmm(query, key.transpose(-2, -1)) / self.head_dim_scale
        attn += attn_mask if attn_mask is not None else 0
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn = torch.bmm(attn, value)
        attn = attn.transpose(0, 1).reshape(tgt_len, batch_size, self.embed_dim)
        attn = self.out_proj(attn).transpose(1, 0)
        return attn, None
