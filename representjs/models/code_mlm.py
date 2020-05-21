import torch
from torch import nn

from models.encoder import CodeEncoder


class CodeMLM(nn.Module):
    def __init__(self, n_tokens, d_model=512, pad_id=None, **encoder_args):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.encoder = CodeEncoder(n_tokens, project=False, pad_id=pad_id, d_model=d_model, **encoder_args)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))

    def forward(self, im):
        features = self.encoder(im)  # L x B x D
        L, B, D = features.shape
        assert D == self.d_model
        features = self.head(features).view(L, B, D)  # L x B x D
        logits = torch.matmul(features, self.encoder.embedding.weight.transpose(0, 1)).view(L, B, self.n_tokens)  # [L, B, ntok]
        return torch.transpose(logits, 0, 1).view(B, L, self.n_tokens)  # [B, T, ntok]
