import math

import torch
import torch.nn as nn

from models.encoder import CodeEncoder, CodeEncoderLSTM


class TypeTransformer(nn.Module):
    def __init__(
        self,
        n_tokens,
        n_output_tokens,
        d_model=512,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        encoder_type="transformer"
    ):
        super(TypeTransformer, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}

        # Encoder
        assert (encoder_type in ["transformer", "lstm"])
        if encoder_type == "transformer":
            self.encoder = CodeEncoder(
                n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False
            )
            # Output for type prediction
            # TODO: Try LeakyReLU
            self.output = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_output_tokens))
        elif encoder_type == "lstm":
            self.encoder = CodeEncoderLSTM(
                n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False
            )
            # Output for type prediction
            # TODO: Try LeakyReLU
            self.output = nn.Sequential(nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, n_output_tokens))

        # Feature aggregation
        # self.lam = nn.Parameter(torch.ones(1, 1, d_model, dtype=torch.float))

    def forward(self, src_tok_ids, output_attention):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            output_attention: [B, L, L] float tensor
        """
        if src_tok_ids.size(0) != output_attention.size(0):
            raise RuntimeError("the batch number of src_tok_ids and output_attention must be equal")

        # Encode
        memory = self.encoder(src_tok_ids)  # LxBxD
        memory = memory.transpose(0, 1)  # BxLxD

        # Aggregate features to the starting token in each labeled identifier
        # memory = memory + self.lam * torch.matmul(output_attention, memory)  # BxLxD
        memory = torch.matmul(output_attention, memory)  # BxLxD

        # Predict types
        logits = self.output(memory)  # BxLxV
        return logits
