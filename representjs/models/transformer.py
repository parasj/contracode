import math

import torch
import torch.nn as nn
from torch.nn import Transformer


class PositionalEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
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
    def __init__(self, ntoken, ninp, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.ntoken = ntoken
        self.ninp = ninp
        self.src_pos_encoder = PositionalEncoding(ninp, dropout, max_len=2048)
        self.tgt_pos_encoder = PositionalEncoding(ninp, dropout, max_len=1024)
        self.transformer = Transformer(d_model=ninp, dropout=dropout)
        self.emb = nn.Embedding(ntoken, ninp)

    def forward(self, src_tok_ids, tgt_tok_ids, pad_id: int):
        """
        Arguments:
            src_tok_ids: [B, L] long tensor
            tgt_tok_ids: [B, T] long tensor
            pad_id: If supplied, we generate a mask for the source and target
        """
        src_emb = self.emb(src_tok_ids).transpose(0, 1) * math.sqrt(self.ninp)
        tgt_emb = self.emb(tgt_tok_ids).transpose(0, 1) * math.sqrt(self.ninp)
        src_emb = self.src_pos_encoder(src_emb)
        tgt_emb = self.tgt_pos_encoder(tgt_emb)

        src_key_padding_mask = src_tok_ids == pad_id
        tgt_key_padding_mask = tgt_tok_ids == pad_id
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_tok_ids.size(1)).to(src_tok_ids.device)
        
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        logits = torch.matmul(output, self.emb.weight.transpose(0, 1))  # [T, B, ntok]
        logits = torch.transpose(logits, 0, 1)  # [B, T, ntok]

        return logits
