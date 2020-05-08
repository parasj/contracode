import pprint
import random

import fire
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
import tqdm
import wandb
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from data.csn_js_pyloader import ComposeTransform, WindowLineCropTransform, CanonicalizeKeysTransform, NumericalizeTransform, \
    AugmentedJSDataset, PadCollateWrapper
from representjs import RUN_DIR, CSNJS_DIR
from representjs.data.csn_js_jsonl import get_csnjs_dataset
from representjs.models.code_moco import CodeMoCo
from representjs.utils import accuracy, count_parameters

CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


class ContrastiveTrainer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            pad_id: int):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        logger.info("Running with configuration:\n{}".format(pprint.pformat(self.config)))
        assert pad_id is not None
        self.pad_id = pad_id
        self.moco_model = CodeMoCo(vocab_size, pad_id=self.pad_id)
        self.config.update(self.moco_model.config)

    def forward(self, imgs_query, imgs_key):
        return self.moco_model(imgs_query, imgs_key)


def training_step(model, batch, use_cuda=False):
    imgs, _ = batch
    if use_cuda:
        imgs = imgs.cuda()
    imgs_k, imgs_q = imgs[:, 0, :], imgs[:, 1, :]
    output, target = model(imgs_q, imgs_k)
    loss = F.cross_entropy(output, target)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    logs = {'pretrain/loss': loss.item(), 'pretrain/acc@1': acc1[0].item(),
            'pretrain/acc@5': acc5[0].item(), 'pretrain/queue_ptr': model.module.moco_model.queue_ptr.item()}
    return {'loss': loss, 'log': logs}


def pretrain(
        run_name: str,

        # Data
        train_filepath: str = CSNJS_TRAIN_FILEPATH,
        spm_filepath: str = SPM_UNIGRAM_FILEPATH,
        program_mode="identity",
        num_workers=1,
        limit_dataset_size=-1,
        max_sequence_length=1024,
        augment_window_crop_size=6,

        # Optimization
        train_decoder_only: bool = False,
        num_epochs: int = 100,
        save_every: int = 2,
        batch_size: int = 256,
        lr: float = 8e-4,
        adam_betas=(0.9, 0.98),

        # Loss
        subword_regularization_alpha: float = 0,

        # Computational
        use_cuda: bool = True,
        seed: int = 0
):
    config = locals()
    logger.info(f"Config: {config}")

    assert not use_cuda or torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = RUN_DIR / run_name
    run_dir.mkdir(exist_ok=True, parents=True)
    logger.add(str((run_dir / "train.log").resolve()))
    logger.info(f"Saving logs, model checkpoints to {run_dir}")
    wandb.init(name=run_name, config=config, job_type="training", project="moco-pretrain", entity="ml4code")

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")

    # Create training dataset and dataloader
    train_dataset = get_csnjs_dataset(train_filepath, label_mode="none", limit_size=limit_dataset_size)
    test_transforms = ComposeTransform([
        WindowLineCropTransform(augment_window_crop_size),
        NumericalizeTransform(SPM_UNIGRAM_FILEPATH, subword_regularization_alpha, max_sequence_length),
        CanonicalizeKeysTransform(data='function'),
    ])
    augmented_dataset = AugmentedJSDataset(train_dataset, test_transforms, contrastive=True)
    collate_wrapper = PadCollateWrapper(contrastive=True, pad_id=pad_id)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_wrapper, num_workers=num_workers,
                              drop_last=True)

    # Create model
    model = ContrastiveTrainer(vocab_size=sp.GetPieceSize(), pad_id=pad_id)
    logger.info(f"Created ContrastiveTrainer with {count_parameters(model)} params")
    model = nn.DataParallel(model)
    model = model.cuda() if use_cuda else model
    params = model.decoder.parameters() if train_decoder_only else model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=1e-9)

    global_step = 0
    min_eval_loss = float("inf")
    for epoch in tqdm.trange(1, num_epochs + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            train_metrics = training_step(model, batch, use_cuda=use_cuda)
            loss = train_metrics["loss"]
            loss.backward()
            optimizer.step()

            # Log loss
            global_step += 1
            wandb.log(dict(epoch=epoch, **train_metrics["log"]), step=global_step)
            pbar.set_description(f"epoch {epoch} loss {loss.item():.4f}")

        # Save checkpoint
        if save_every and epoch % save_every == 0:
            checkpoint = {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            }
            model_file = run_dir / f"ckpt_pretrain_ep{epoch:04d}.pth"
            logger.info(f"Saving checkpoint to {model_file}...")
            torch.save(checkpoint, str(model_file.resolve()))
            logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(pretrain)
