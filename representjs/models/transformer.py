import math

import torch
import torch.nn as nn

from models.code_moco import CodeEncoder


class TransformerModel(nn.Module):
    def __init__(self, n_tokens, d_model=512, d_rep=128, n_head=8, n_encoder_layers=6, d_ff=2048, dropout=0.1,
                 activation="relu", norm=True, pad_id=None, n_decoder_layers=6):
        super(TransformerModel, self).__init__()
        self.config = {k: v for k, v in locals().items() if k != 'self'}

        # Encoder
        self.encoder = CodeEncoder(n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, d_ff, dropout, activation)
        assert norm
        decoder_norm = nn.LayerNorm(d_model) if norm else None
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers, norm=decoder_norm)

        assert pad_id != None

    def forward(self, src_tok_ids, tgt_tok_ids):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            tgt_tok_ids: [B, T] long tensor
        """
        if src_tok_ids.size(0) != tgt_tok_ids.size(0):
            raise RuntimeError("the batch number of src_tok_ids and tgt_tok_ids must be equal")

        # Encode
        memory = self.encoder(src_tok_ids)

        # Decode, using the same embedding and positional encoding as the encoder
        tgt_emb = self.encoder.embedding(tgt_tok_ids).transpose(0, 1) * math.sqrt(self.config['d_model'])
        tgt_emb = self.encoder.pos_encoder(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(tgt_tok_ids.size(1)).to(tgt_tok_ids.device)
        if self.config['pad_id'] is None:
            assert False
            tgt_key_padding_mask = None
            memory_key_padding_mask = None
        else:
            tgt_key_padding_mask = tgt_tok_ids == self.config['pad_id']
            # memory_key_padding_mask = src_tok_ids == self.config['pad_id']
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None,
                              tgt_key_padding_mask=tgt_key_padding_mask)
                            #   memory_key_padding_mask=memory_key_padding_mask)

        logits = torch.matmul(output, self.encoder.embedding.weight.transpose(0, 1))  # [T, B, ntok]
        return torch.transpose(logits, 0, 1)  # [B, T, ntok]

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
