import torch
import torch.nn as nn

from models.encoder import CodeEncoder, CodeEncoderLSTM


class TypeTransformer(nn.Module):
    def __init__(
        self,
        n_tokens,
        n_output_tokens,
        d_model=512,
        d_out_projection=512,
        n_hidden_output=1,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        encoder_type="transformer",
    ):
        super(TypeTransformer, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}

        # Encoder and output for type prediction
        assert encoder_type in ["transformer", "lstm"]
        if encoder_type == "transformer":
            self.encoder = CodeEncoder(
                n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False
            )
            # TODO: Try LeakyReLU
            self.output = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_output_tokens))
        elif encoder_type == "lstm":
            self.encoder = CodeEncoderLSTM(
                n_tokens=n_tokens,
                d_model=d_model,
                d_rep=d_rep,
                n_encoder_layers=n_encoder_layers,
                dropout=dropout,
                pad_id=pad_id,
                project=False,
            )
            layers = []
            layers.append(nn.Linear(d_model * 2, d_out_projection))
            if n_hidden_output > 1:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            for hidden_idx in range(n_hidden_output - 1):
                layers.append(nn.Linear(d_out_projection, d_out_projection))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(d_out_projection, n_output_tokens))
            self.output = nn.Sequential(*layers)

    def forward(self, src_tok_ids, lengths=None, output_attention=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            output_attention: [B, L, L] float tensor
        """
        if output_attention is not None and src_tok_ids.size(0) != output_attention.size(0):
            raise RuntimeError("the batch number of src_tok_ids and output_attention must be equal")

        # Encode
        memory, _ = self.encoder(src_tok_ids, lengths)  # LxBxD
        memory = memory.transpose(0, 1)  # BxLxD

        if output_attention is not None:
            # Aggregate features to the starting token in each labeled identifier
            memory = torch.matmul(output_attention, memory)  # BxLxD

        # Predict logits over types
        return self.output(memory)  # BxLxV
