import torch
from torch import nn

from models.encoder import CodeEncoder


class CodeMLM(nn.Module):
    def __init__(self, n_tokens, d_model=512, pad_id=None, **encoder_args):
        super().__init__()
        self.encoder = CodeEncoder(n_tokens, project=False, pad_id=pad_id, d_model=d_model, **encoder_args)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, im):
        """
        Input:
            im: a batch of query images
        Output:
            logits
        """
        features = self.encoder(im)  # LxBxD
        features = self.head(features)  # LxBxD
        logits = torch.matmul(features, self.encoder.embedding.weight.transpose(0, 1))  # [L, B, ntok]
        return torch.transpose(logits, 0, 1)  # [B, T, ntok]
