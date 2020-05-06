import math

import fire
import pytorch_lightning as pl
import sentencepiece as spm
import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from data.csn_js import JSONLinesDataset, javascript_dataloader
from models import TransformerModel, PositionalEncoding
from representjs import RUN_DIR, CSNJS_DIR

CSNJS_TRAIN_FILEPATH = CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz"
SPM_UNIGRAM_FILEPATH = CSNJS_DIR / "csnjs_8k_9995p_unigram.model"


class CodeEncoder(nn.Module):
    def __init__(self, n_tokens, d_model=512, n_head=8, n_encoder_layers=6, d_ff=2048, dropout=0.1, activation="relu",
                 norm=True, pad_id=None):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != 'self'}
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.src_pos_encoder = PositionalEncoding(d_model, dropout, max_len=2048)
        norm_fn = nn.LayerNorm(d_model) if norm else None
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)

    def forward(self, x):
        src_emb = self.embedding(x).transpose(0, 1) * math.sqrt(self.config['d_model'])
        src_emb = self.src_pos_encoder(src_emb)
        if self.config['pad_id'] is not None:
            src_key_padding_mask = x == self.config['pad_id']
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_tok_ids.size(1)).to(src_tok_ids.device)

        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)

        logits = torch.matmul(output, self.emb.weight.transpose(0, 1))  # [T, B, ntok]
        logits = torch.transpose(logits, 0, 1)  # [B, T, ntok]

        return logits


class ContrastiveTrainer(pl.LightningModule):
    def __init__(
            self,
            run_name: str,
            n_epochs: int,
            batch_size: int,
            lr: float,
            adam_betas=(0.9, 0.98),
            iters_per_epoch: int = -1,
            subword_regularization_alpha=0.,
            train_ds_path: str = CSNJS_TRAIN_FILEPATH,
            spm_filepath: str = SPM_UNIGRAM_FILEPATH,
            num_workers: int = 8,
            data_limit_size: int = -1):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != 'self'}
        self.config['contrastive_augmentations'] = [{"fn": "sample_lines", "line_length_pct": 0.5}]  # todo
        self.sm = self.load_sentencepiece(spm_filepath)
        self.transformer_model = TransformerModel(ntoken=self.sp.GetPieceSize(), ninp=512)
        self.config.update(self.transformer_model.config)
        self.run_dir = RUN_DIR / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        pass

    @staticmethod
    def load_sentencepiece(spm_filename):
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_filename)
        return sp

    def train_dataloader(self):
        dataset_fields = {"function": "function"}
        dataset_require_fields = []
        train_dataset = JSONLinesDataset(self.config['train_ds_path'], fields=dataset_fields,
                                         require_fields=dataset_require_fields, limit_size=self.config['data_limit_size'])
        logger.info("Training dataset size:", len(train_dataset))
        train_loader = javascript_dataloader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True,
            num_workers=self.data_loader_params['num_workers'],
            augmentations=self.config['contrastive_augmentations'], sp=self.sm, program_mode='contrastive',
            subword_regularization_alpha=self.config['subword_regularization_alpha'])
        return train_loader

    def configure_optimizers(self):  # todo scheduler
        return [torch.optim.Adam(self.parameters(), lr=self.config['lr'], betas=self.config['adam_betas'])]

    def fit(self):
        wandb_logger = WandbLogger(entity="ml4code", project="code-representation", name=self.run_name, log_model=True)
        wandb_logger.watch(self, log="all")
        wandb_logger.log_hyperparams(self.config)
        trainer = Trainer(logger=wandb_logger, default_root_dir=self.run_dir)
        trainer.fit(self)


if __name__ == "__main__":
    fire.Fire(ContrastiveTrainer)
