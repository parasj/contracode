import math

import torch
import torch.nn as nn

from models.encoder import CodeEncoder, CodeEncoderLSTM

from loguru import logger


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        n_decoder_layers=6,
    ):
        super(TransformerModel, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}

        # Encoder
        self.encoder = CodeEncoder(
            n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, d_ff, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model) if norm else None
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers, norm=decoder_norm)

    def encode(self, src_tok_ids, src_lengths=None):
        # print("transformer.py encode src_tok_ids", src_tok_ids.shape)
        memory, _ = self.encoder(src_tok_ids)  # [T_src, B, d_model]
        # print("transformer.py encode memory", memory.shape)
        return memory

    def decode_no_masking(self, memory, tgt_tok_ids):
        tgt_emb = self.encoder.embedding(tgt_tok_ids).transpose(0, 1) * math.sqrt(self.config["d_model"])
        tgt_emb = self.encoder.pos_encoder(tgt_emb)
        output = self.decoder(tgt_emb, memory)
        logits = torch.matmul(output, self.encoder.embedding.weight.transpose(0, 1))  # [T, B, ntok]
        return torch.transpose(logits, 0, 1)  # [B, T_tgt, ntok]

    def decode(self, memory, tgt_tok_ids, tgt_lengths=None):
        tgt_emb = self.encoder.embedding(tgt_tok_ids).transpose(0, 1) * math.sqrt(self.config["d_model"])
        tgt_emb = self.encoder.pos_encoder(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(tgt_tok_ids.size(1)).to(tgt_tok_ids.device)
        tgt_key_padding_mask = tgt_tok_ids == self.config["pad_id"]
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = torch.matmul(output, self.encoder.embedding.weight.transpose(0, 1))  # [T, B, ntok]
        return torch.transpose(logits, 0, 1)  # [B, T_tgt, ntok]

    def forward(self, src_tok_ids, tgt_tok_ids, src_lengths=None, tgt_lengths=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            tgt_tok_ids: [B, T] long tensor
        """
        if src_tok_ids.size(0) != tgt_tok_ids.size(0):
            raise RuntimeError("the batch number of src_tok_ids and tgt_tok_ids must be equal")

        memory = self.encode(src_tok_ids, src_lengths)
        logits = self.decode(memory, tgt_tok_ids, tgt_lengths)  # [B, T, ntok]
        return logits

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self, n_tokens, d_model=512, n_encoder_layers=2, dropout=0.1, activation="relu", norm=True, pad_id=None,
    ):
        super(Seq2SeqLSTM, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}
        assert self.config["pad_id"] is not None

        d_rep = d_model  # so same embedding can be used by encoder and decoder

        # Encoder
        self.encoder = CodeEncoderLSTM(
            n_tokens=n_tokens,
            d_model=d_model,
            d_rep=d_rep,
            n_encoder_layers=n_encoder_layers,
            dropout=dropout,
            pad_id=pad_id,
            project="hidden",
        )

        # Decoder
        self.decoder = nn.LSTM(input_size=d_model, hidden_size=d_rep, num_layers=1, bidirectional=False, dropout=dropout)
        self.decoder_c_0 = nn.Parameter(torch.zeros(1, 1, d_rep))
        # self.decoder_proj = nn.Sequential(nn.Linear(d_rep, d_model), nn.ReLU())

    def forward(self, src_tok_ids, tgt_tok_ids, src_lengths, tgt_lengths):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            tgt_tok_ids: [B, T] long tensor
        """
        if src_tok_ids.size(0) != tgt_tok_ids.size(0):
            raise RuntimeError("the batch number of src_tok_ids and tgt_tok_ids must be equal")

        # Encode
        oh_0 = self.encoder(src_tok_ids, src_lengths)  # B x d_rep
        oh_0 = oh_0.unsqueeze(0)  # 1 x B x d_rep

        # Decode, using the same embedding as the encoder
        # TODO: Try a different subword vocab, or a non-subword vocab
        tgt_emb = self.encoder.embedding(tgt_tok_ids).transpose(0, 1) * math.sqrt(self.config["d_model"])
        tgt_emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            tgt_emb, tgt_lengths - 1, enforce_sorted=False
        )  # subtract 1 from lengths since targets are expected to be shifted
        output, _ = self.decoder(tgt_emb_packed, (oh_0, self.decoder_c_0.expand_as(oh_0)))  # [T, B, d_rep] (packed)
        # output = self.decoder_proj(output)  # [T, B, d_model] (packed)
        # print("Prior to pading output, shapes:")
        # print("oh_0.shape", oh_0.shape)
        # print("src_tok_ids.shape", src_tok_ids.shape)
        # print("tgt_tok_ids.shape", tgt_tok_ids.shape)
        # print("src_lengths.shape", src_lengths.shape)
        # print("src_length min", src_lengths.min())
        # print("src_length max", src_lengths.max())
        # print("tgt_lengths.shape", tgt_lengths.shape)
        # print("tgt_length min", tgt_lengths.min())
        # print("tgt_length max", tgt_lengths.max())
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=tgt_tok_ids.size(1))  # [B, T, d_model]
        # print("After packing", output.shape)
        logits = torch.matmul(output, self.encoder.embedding.weight.transpose(0, 1))  # [B, T, ntok]
        return logits
