import math

from torch import nn

from representjs.models.moco_template import MoCoTemplate
from models.positional_embedding import PositionalEncoding


class CodeMoCo(MoCoTemplate):
    def __init__(self, n_tokens, d_model=512, d_rep=128, K=65536, m=.999, T=0.07, encoder_config={}, pad_id=None):
        super().__init__(d_rep, K, m, T, dict(n_tokens=n_tokens, d_model=d_model, d_rep=d_rep, pad_id=pad_id, **encoder_config))

    def make_encoder(self, n_tokens, d_model, d_rep, pad_id=None, **kwargs):
        return CodeEncoder(n_tokens, project=True, pad_id=pad_id, d_model=d_model, d_rep=d_rep, **kwargs)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        return super().forward(im_q, im_k)


class CodeEncoder(nn.Module):
    def __init__(self, n_tokens, d_model=512, d_rep=256, n_head=8, n_encoder_layers=6, d_ff=2048, dropout=0.1,
                 activation="relu", norm=True, pad_id=None, project=False):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != 'self'}
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=2048)
        norm_fn = nn.LayerNorm(d_model) if norm else None
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)
        if project:
            self.project_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))

        # NOTE: We use the default PyTorch intialization, so no need to reset parameters.

    def forward(self, x):
        src_emb = self.embedding(x).transpose(0, 1) * math.sqrt(self.config['d_model'])
        src_emb = self.pos_encoder(src_emb)
        if self.config['pad_id'] is not None:
            src_key_padding_mask = x == self.config['pad_id']
        else:
            src_key_padding_mask = None
        out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)  # TxBxD
        if self.config['project']:
            return self.project_layer(out.mean(dim=0))
        else:
            return out
