import os
import time
import pathlib
import pprint

import fire
import pytorch_lightning as pl
import sentencepiece as spm
import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from data.csn_js.jsonl_dataset import JSONLinesDataset
from representjs.data.csn_js_pyloader import AugmentedJSDataset, PadCollateWrapper
from data.transforms import NumericalizeTransform, WindowLineCropTransform, CanonicalizeKeysTransform, ComposeTransform
from representjs.models.code_moco import CodeMoCo
from representjs import RUN_DIR, CSNJS_DIR
from representjs.utils import accuracy

CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


class ContrastiveTrainer(pl.LightningModule):
    def __init__(
            self,
            n_epochs: int,
            batch_size: int,
            lr: float,
            adam_betas=(0.9, 0.98),
            weight_decay: float = 0.,
            subword_regularization_alpha=0.,
            train_ds_path: str = CSNJS_TRAIN_FILEPATH,
            spm_filepath: str = SPM_UNIGRAM_FILEPATH,
            num_workers: int = 8,
            data_limit_size: int = -1,
            max_length: int = 1024):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        logger.info("Running with configuration:\n{}".format(pprint.pformat(self.config)))
        sm = self.load_sentencepiece(self.config['spm_filepath'])
        self.pad_id = sm.PieceToId("[PAD]") 
        self.moco_model = CodeMoCo(sm.GetPieceSize(), pad_id=self.pad_id)
        self.config.update(self.moco_model.config)

    def forward(self, imgs_query, imgs_key):
        return self.moco_model(imgs_query, imgs_key)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs_k, imgs_q = imgs[:, 0, :], imgs[:, 1, :]
        output, target = self(imgs_q, imgs_k)
        loss = F.cross_entropy(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        logs = {'pretrain_loss': loss, 'pretrain_acc@1': acc1[0],
                'pretrain_acc@5': acc5[0], 'pretrain_queue_ptr': self.moco_model.queue_ptr}
        return {'loss': loss, 'log': logs}

    @staticmethod
    def load_sentencepiece(spm_filename):
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_filename)
        return sp

    def configure_optimizers(self):  # todo scheduler
        return [torch.optim.Adam(self.parameters(), lr=self.config['lr'], betas=self.config['adam_betas'],
                                 weight_decay=self.config['weight_decay'])]

    def train_dataloader(self):
        dataset_fields = {"function": "function"}
        jsonl_dataset = JSONLinesDataset(self.config['train_ds_path'], fields=dataset_fields,
                                         require_fields=[], limit_size=self.config['data_limit_size'])

        train_dataset = AugmentedJSDataset(jsonl_dataset, self.make_transforms(), contrastive=True, max_length=self.config['max_length'])
        logger.info("Training dataset size:", len(train_dataset))
        collate_wrapper = PadCollateWrapper(contrastive=True, pad_id=self.pad_id)
        return DataLoader(train_dataset, self.config['batch_size'], shuffle=True,
                          collate_fn=collate_wrapper,
                          num_workers=self.config['num_workers'],
                          drop_last=True)

    def make_transforms(self):
        return ComposeTransform([
            WindowLineCropTransform(6),
            NumericalizeTransform(self.config['spm_filepath'], self.config['subword_regularization_alpha'],
                                  self.config['max_length']),
            CanonicalizeKeysTransform(data='function_ids')])


def fit(n_epochs: int, run_name: str, use_gpu=True, use_fp16=False, run_dir_base=RUN_DIR,
        dataset=CSNJS_TRAIN_FILEPATH, vocab=SPM_UNIGRAM_FILEPATH, **kwargs):
    setattr(WandbLogger, 'name', property(lambda self: self._name))
    extra_trainer_args = dict()
    if use_fp16:
        extra_trainer_args.update(dict(amp_level='O1', precision=16))
    logger.info("Using extra training arguments for Pytorch Lightning {}".format(extra_trainer_args))
    logger.info("Training model with run name {}, use_gpu = {}".format(run_name, use_gpu))

    run_dir = (pathlib.Path(run_dir_base) / "{}_{}".format(run_name, int(time.time()))).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving results to {}".format(run_dir))
    
    model = ContrastiveTrainer(n_epochs=n_epochs, train_ds_path=dataset, spm_filepath=vocab, **kwargs)
    wandb_logger = WandbLogger(name=run_name, save_dir=str(run_dir), entity="ml4code", project="code-representation", log_model=True)
    # wandb_logger.watch(model, log="all")
    wandb_logger.log_hyperparams(model.config)

    checkpoint_callback = ModelCheckpoint(
        filepath=str(run_dir) + "/weights_{epoch:04d}.ckpt",
        verbose=True,
        period=1,
        save_top_k=-1
    )

    trainer = Trainer(logger=wandb_logger, default_root_dir=str(run_dir), benchmark=False,
                      distributed_backend="dp", gpus=-1 if use_gpu else None, max_epochs=n_epochs, checkpoint_callback=checkpoint_callback,
                      **extra_trainer_args)
    logger.info("CUDA_DEVICE_ORDER={}".format(os.environ.get("CUDA_DEVICE_ORDER")))
    logger.info("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES")))
    logger.info("trainer.is_slurm_managing_tasks = {}".format(trainer.is_slurm_managing_tasks))
    
    trainer.fit(model)


if __name__ == "__main__":
    fire.Fire(fit)