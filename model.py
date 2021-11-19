import math

import torch
from torch import nn
# from torch.nn import Transformer

# PyTorch compatible Transformer implementation (within this example)
from lib.nn import Transformer


__all__ = [
    "PositionalEncoding",
    "TransformerModel",
]


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, step=2) * -(math.log(10000) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_model_scale = math.sqrt(d_model)
        self.embedder = nn.Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                       dim_feedforward, dropout, batch_first=True)
        self.out_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.embedder(src) * self.d_model_scale
        tgt = self.embedder(tgt) * self.d_model_scale
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        out = self.transformer(
            src, tgt, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        out = self.out_linear(out)
        return out
