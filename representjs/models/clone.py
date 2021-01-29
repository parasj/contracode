import torch
import torch.nn as nn

from models.encoder import CodeEncoder, CodeEncoderLSTM


class CloneModel(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        encoder_type="transformer",
        critic_type="bilinear_identity",
        bilinear_rank=None,
    ):
        super().__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}

        # Encoder and output for type prediction
        assert encoder_type in ["transformer", "lstm"]
        if encoder_type == "transformer":
            d_critic_rep = d_model  # Per token dimension, then take mean
            self.encoder = CodeEncoder(
                n_tokens=n_tokens,
                d_model=d_model,
                n_head=n_head,
                n_encoder_layers=n_encoder_layers,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm=norm,
                pad_id=pad_id,
                project=False,
            )
        elif encoder_type == "lstm":
            d_critic_rep = 4 * d_model  # 4 * d_model for 2 layer bidirectional LSTM
            self.encoder = CodeEncoderLSTM(
                n_tokens=n_tokens, d_model=d_model, n_encoder_layers=n_encoder_layers, dropout=dropout, pad_id=pad_id, project=False,
            )

        if critic_type == "bilinear_diagonal":
            self.output_weight = nn.Parameter(torch.randn(d_critic_rep), requires_grad=True)
        elif critic_type == "bilinear_symmetric":
            self.output_weight = nn.Parameter(torch.randn(d_critic_rep, d_critic_rep), requires_grad=True)
        elif critic_type == "bilinear_symmetric_plus_identity":
            W = torch.randn(d_critic_rep, d_critic_rep) + torch.eye(d_critic_rep)
            self.output_weight = nn.Parameter(W, requires_grad=True)
        elif critic_type == "bilinear_identity":
            self.output_weight = None
        elif critic_type == "bilinear_lowrank":
            assert bilinear_rank
            W = torch.randn(bilinear_rank, d_critic_rep)
            self.output_weight = nn.Parameter(W, requires_grad=True)
        else:
            raise ValueError

    def parameters(self):
        return [self.output_weight]

    def output(self, rep):
        """
        Args:
            rep: [2, B, dim]
        Returns:
            similarity: [B]
        """
        if self.config["critic_type"] == "bilinear_identity":  # cosine similarity
            rep = nn.functional.normalize(rep, dim=-1)
            sim = torch.sum(rep[0] * rep[1], dim=-1)
        elif self.config["critic_type"] == "bilinear_diagonal":
            rep = nn.functional.normalize(rep, dim=-1)
            sim = torch.sum(rep[0] * self.output_weight.unsqueeze(0) * rep[1], dim=-1)
        elif self.config["critic_type"].startswith("bilinear_symmetric"):
            rep = nn.functional.normalize(rep, dim=-1)
            # Symmetrize weight matrix for bilinear form
            W = 0.5 * (self.output_weight + self.output_weight.T)
            sim = torch.sum(torch.matmul(rep[0], W) * rep[1], dim=-1)
        elif self.config["critic_type"] == "bilinear_lowrank":
            # Project each representation followed by cosine similarity
            # sim = (Wa)^T(Wb) / |Wa||Wb|
            #     = a^T W^T W b / |Wa||Wb|
            # W = torch.mm(self.output_weight.T, self.output_weight)
            # sim = torch.sum(torch.matmul(rep[0], W) * rep[1], dim=-1)

            # rep0 = torch.einsum("bd,dr->br", rep[0], self.output_weight.T)
            rep0 = torch.mm(rep[0], self.output_weight.T)
            rep0 = nn.functional.normalize(rep0, dim=-1)

            # rep1 = torch.einsum("bd,dr->br", rep[1], self.output_weight.T)
            rep1 = torch.mm(rep[1], self.output_weight.T)
            rep1 = nn.functional.normalize(rep1, dim=-1)

            sim = torch.sum(rep0 * rep1, dim=-1)

        assert sim.ndim == 1
        assert sim.size(0) == rep.size(1)

        return sim

    def forward(self, src_tok_ids, lengths=None):
        r"""
        Arguments:
            src_tok_ids: [2B, L] long tensor
            lengths: [2B] long tensor
        """
        # print("CloneModel: src_tok_ids shape", src_tok_ids.shape)
        # if lengths is not None:
        #     print("CloneModel: lengths shape", lengths.shape)

        # Encode
        with torch.no_grad():
            memory, h_n = self.encoder(src_tok_ids, lengths)  # [L, 2B, D], [4, 2B, D]
            if h_n is not None:
                rep = torch.flatten(h_n.transpose(0, 1), start_dim=1)  # [2B, 4D]
            else:
                rep = memory.mean(dim=0)  # [2B, D]

            # Reshape to [2, B, dim]
            rep = rep.view(2, rep.size(0) // 2, rep.size(1))

        return self.output(rep)  # [B]
