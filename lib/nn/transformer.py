from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
# from torch.nn import MultiheadAttention

# PyTorch compatible MultiheadAttention implementation (within this example)
from .activation import MultiheadAttention


__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
]


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,  # Dummy parameter; always set to True
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        # Encoder module
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # Decoder module
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        # Reset parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, batch_first=True):
        super().__init__()
        # Attention modules
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Feed-forward modules
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Encoder modules
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feed-forward
        src2 = self.linear1(src)
        src2 = F.relu(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, batch_first=True):
        super().__init__()
        # Attention modules
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Feed-forward modules
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Decoder modules
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Cross-attention
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # Feed-forward
        tgt2 = self.linear1(tgt)
        tgt2 = F.relu(tgt2)
        tgt2 = self.dropout(tgt2)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
