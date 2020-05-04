import fire
import pytorch_lightning as pl
import sentencepiece as spm
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from models import TransformerModel
from representjs import RUN_DIR
from representjs.data.csn_js import CSNJS_TRAIN_FILEPATH, SPM_FILEPATH, JSONLinesDataset, javascript_dataloader


class ContrastiveModel(pl.LightningModule):
    def __init__(
            self,
            # data loader
            train_ds_path: str = CSNJS_TRAIN_FILEPATH,
            spm_filepath: str = SPM_FILEPATH,
            num_workers: int = 4,
            data_limit_size: int = -1,

            # train params
            batch_size: int = 256,
            lr: float = 8e-4,
            lr_warmup_percent = 0.02,
            adam_betas = (0.9, 0.98),
            n_epochs: int = 100,

            # hyperparams
            subword_regularization_alpha: float = 0,
    ):
        super().__init__()
        self.config = locals()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_filepath)
        self.PAD_ID = self.sp.PieceToId("[PAD]")

        self.contrastive_augmentations = [{"fn": "sample_lines", "line_length_pct": 0.5}]
        self.transformer_model = TransformerModel(ntoken=self.sp.GetPieceSize(), ninp=512)
        self.

    def forward(self, x):
        return self.transformer_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.config['lr'],
                                 betas=self.config['adam_betas'], eps=self.config['lr'])
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            self.config['lr'],
            epochs=self.config['n_epochs'],
            steps_per_epoch=len(self.train_dataloader()),
            pct_start=self.config['lr_warmup_percent'])
        return [optim], [sched]

    def train_dataloader(self):
        dataset_fields = {"function": "function"}
        dataset_require_fields = []
        train_dataset = JSONLinesDataset(self.config['train_ds_path'], fields=dataset_fields,
                                         require_fields=dataset_require_fields, limit_size=self.limit_dataset_size)
        print("Training dataset size:", len(train_dataset))
        train_loader = javascript_dataloader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'],
            augmentations=self.contrastive_augmentations, sp=self.sp, program_mode="contrastive",
            subword_regularization_alpha=self.config['subword_regularization_alpha'])
        return train_loader


def train(run_name: str):
    run_dir = RUN_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(entity="ml4code", project="code-representation", name=run_name)
    trainer = Trainer(logger=wandb_logger, default_root_dir=run_dir)
    model = ContrastiveModel(run_name)
    wandb_logger.watch(model)
    wandb_logger.log_hyperparams(model.config)
    trainer.fit(model)


if __name__ == "__main__":
    fire.Fire({'train': train})
