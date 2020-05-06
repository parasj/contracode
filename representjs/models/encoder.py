import math

from torch import nn

from models.transformer import PositionalEncoding


class CodeEncoder(nn.Module):
    def __init__(self, n_tokens, d_model=512, d_rep=128, n_head=8, n_encoder_layers=6, d_ff=2048, dropout=0.1,
                 activation="relu",
                 norm=True, pad_id=None, project=False):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != 'self'}
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.src_pos_encoder = PositionalEncoding(d_model, dropout, max_len=2048)
        norm_fn = nn.LayerNorm(d_model) if norm else None
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)
        if project:
            self.project_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))

    def forward(self, x):
        src_emb = self.embedding(x).transpose(0, 1) * math.sqrt(self.config['d_model'])
        src_emb = self.src_pos_encoder(src_emb)
        if self.config['pad_id'] is not None:
            src_key_padding_mask = x == self.config['pad_id']
        else:
            src_key_padding_mask = None
        out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)  # TxBxD
        if self.config['project']:
            return self.project_layer(out.mean(dim=0))
        else:
            return out