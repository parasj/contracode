import torch
from torch import nn

from models.encoder import CodeEncoder, CodeEncoderLSTM
from models.code_moco import CodeMoCo


class CodeMLM(nn.Module):
    def __init__(self, n_tokens, d_model=512, pad_id=None, encoder_type="transformer", **encoder_args):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        if encoder_type == "transformer":
            self.encoder = CodeEncoder(n_tokens, project=False, pad_id=pad_id, d_model=d_model, **encoder_args)
            self.head_in = d_model
        elif encoder_type == "lstm":
            self.encoder = CodeEncoderLSTM(
                n_tokens=n_tokens,
                d_model=d_model,
                pad_id=pad_id,
                project=False,
                **encoder_args
            )
            self.head_in = 2 * d_model
        else:
            raise ValueError

        self.head = nn.Sequential(nn.Linear(self.head_in, d_model), nn.ReLU(), nn.LayerNorm(d_model))

    def embed(self, im):
        return self.encoder(im)
        
    def forward(self, im, lengths):
        features = self.encoder(im, lengths)  # L x B x D=head_in
        L, B, D = features.shape
        if D != self.head_in:
            print("WARNING: CodeMLM size mismatch.")
            print("self.d_model", self.d_model)
            print("self.head_in", self.head_in)
            print("features shape", features.shape)
            print("L B D", L, B, D)
            print("im shape", im.shape)
            print("lengths shape", lengths.shape)
        features = self.head(features).view(L, B, self.d_model)  # L x B x D=d_model
        logits = torch.matmul(features, self.encoder.embedding.weight.transpose(0, 1)).view(L, B, self.n_tokens)  # [L, B, ntok]
        return torch.transpose(logits, 0, 1).view(B, L, self.n_tokens)  # [B, T, ntok]


class CodeContrastiveMLM(CodeMoCo):
    def __init__(self, n_tokens, d_model=512, d_rep=128, K=61440, m=0.999, T=0.07, pad_id=0, encoder_args={}):
        super().__init__(n_tokens, d_model=d_model, d_rep=d_rep, K=K, m=m, T=T, pad_id=pad_id)
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.d_rep = d_rep
        self.mlm_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))

    def mlm_forward(self, im):  # predicted masked tokens
        features = self.encoder_q(im, no_project_override=True)  # L x B x D
        assert len(features.shape) == 3, str(features.shape)
        L, B, D = features.shape
        assert D == self.d_model
        features = self.mlm_head(features).view(L, B, D)  # L x B x D
        logits = torch.matmul(features, self.encoder_q.embedding.weight.transpose(0, 1)).view(L, B, self.n_tokens)  # [L, B, ntok]
        return torch.transpose(logits, 0, 1).view(B, L, self.n_tokens)  # [B, T, ntok]

    def moco_forward(self, im_q, im_k):  # logits, labels
        return super().forward(im_q, im_k)

    def forward(self, im_q, im_k):
        predicted_masked_tokens = self.mlm_forward(im_q)
        moco_logits, moco_targets = self.moco_forward(im_q, im_k)
        return predicted_masked_tokens, moco_logits, moco_targets
