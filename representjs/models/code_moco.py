from models.encoder import CodeEncoder
from models.moco_template import MoCoTemplate


class CodeMoCo(MoCoTemplate):
    def __init__(self, n_tokens, d_model=512, d_rep=128, K=65536, m=.999, T=0.07, encoder_config={}, pad_id=None):
        super().__init__(d_rep, K, m, T)
        self.config.update({'n_tokens': n_tokens, 'd_model': d_model, 'd_rep': d_rep})
        self.config.update(encoder_config)
        self.pad_id = pad_id
        self.encoder_config = encoder_config

    def make_encoder(self):
        return CodeEncoder(self.config['n_tokens'], project=True, pad_id=self.pad_id, d_model=self.config['d_model'],
                           d_rep=self.config['d_rep'], **self.encoder_config)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        return super().forward(im_q, im_k)