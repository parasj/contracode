import torch
import torch.nn as nn

from models.encoder import CodeEncoder, CodeEncoderLSTM


class CloneModel(nn.Module):
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
        elif encoder_type == "lstm":
            self.encoder = CodeEncoderLSTM(
                n_tokens=n_tokens,
                d_model=d_model,
                d_rep=d_rep,
                n_encoder_layers=n_encoder_layers,
                dropout=dropout,
                pad_id=pad_id,
                project="hidden_identity",
            )
        self.output = nn.Linear(d_model, 1)

    def forward(self, src_tok_ids, lengths=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            output_attention: [B, L, L] float tensor
        """
        print("CloneModel: src_tok_ids shape", src_tok_ids.shape)
        if lengths:
            print("CloneModel: lengths shape", lengths.shape)

        # Encode
        memory = self.encoder(src_tok_ids, lengths)  # LxBxD
        print("CloneModel: memory shape before transpose", memory.shape)
        memory = memory.transpose(0, 1)  # BxH

        # Predict logits over types
        output = self.output(memory)  # Bx1
        print("CloneModel: output shape", output.shape)
