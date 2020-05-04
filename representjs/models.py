import math

import torch
import torch.nn as nn
from torch.nn import Transformer


class PositionalEncoding(nn.Module):
    """
    From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer()
        self.emb = nn.Embedding(ntoken, ninp)

    def forward(self, src_tok_ids, tgt_tok_ids, batch_first=False, pad_id: int=None):
        """
        Arguments:
            src_tok_ids: [L, B] or [B, L] long tensor
            tgt_tok_ids: [T, B] or [B, T] long tensor
            batch_first: whether batch dimension is before sequence length dimension in inputs
            pad_id: If supplied, we generate a mask for the source and target
        """
        src_emb = self.emb(src_tok_ids)
        tgt_emb = self.emb(tgt_tok_ids)
        if batch_first:
            src_emb = torch.transpose(src_emb, 0, 1)
            tgt_emb = torch.transpose(tgt_emb, 0, 1)
        # TODO: Add positional embedding

        if pad_id is not None:
            # TODO: add src_key_padding_mask, tgt_key_padding_mask
            raise NotImplementedError
        output = self.transformer(src_emb, tgt_emb)

        logits = torch.matmul(output, self.emb.weight.transpose(0, 1))
        if batch_first:
            logits = torch.transpose(logits, 0, 1)
        # print("Input shapes:", src_tok_ids.shape, tgt_tok_ids.shape)
        # print("Embedding shapes:", src_emb.shape, tgt_emb.shape)
        # print("Output shape:", output.shape)
        # print("Logits shape:", logits.shape)

        return logits
