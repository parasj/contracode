import os
import random
import time

import fire
import numpy as np
import sentencepiece as spm
import torch
import tqdm
import wandb
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from representjs import RUN_DIR, CSNJS_DIR
from data.precomputed_dataset import PrecomputedDataset
from models.code_mlm import CodeMLM
from models.code_moco import CodeMoCo
from utils import accuracy, count_parameters

DEFAULT_CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
DEFAULT_SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


def pretrain(
    run_name: str,
    # Data
    train_filepath: str = DEFAULT_CSNJS_TRAIN_FILEPATH,
    spm_filepath: str = DEFAULT_SPM_UNIGRAM_FILEPATH,
    num_workers=1,
    limit_dataset_size=-1,
    max_sequence_length=1024,
    augment_window_crop_size=6,
    subword_regularization_alpha: float = 0,
    program_mode="contrastive",
    loss_mode="infonce",
    # Optimization
    num_epochs: int = 100,
    save_every: int = 1,
    batch_size: int = 256,
    lr: float = 8e-4,
    adam_betas=(0.9, 0.98),
    # Computational
    use_cuda: bool = True,
    seed: int = 0,
):
    run_name = str(run_name)  # support numerical run ids
    slurm_job_id, slurm_job_hostname = (
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_JOB_NODELIST"),
    )
    config = locals()
    logger.info("Training configuration: {}".format(config))
    logger.info(
        "CUDA_VISIBLE_DEVICES = '{}', CUDA_DEVICE_ORDER = '{}'".format(
            os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_DEVICE_ORDER")
        )
    )

    assert not use_cuda or torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"
    assert loss_mode in ["infonce", "mlm"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = RUN_DIR / "{}_{}".format(run_name, int(time.time()))
    run_dir.mkdir(exist_ok=True, parents=True)
    logger.add(str((run_dir / "train.log").resolve()))
    logger.info(f"Saving logs, model checkpoints to {run_dir}")
    wandb.init(
        name=run_name, config=config, job_type="training", project="moco-pretrain", entity="ml4code",
    )

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")

    # Create training dataset and dataloader
    assert train_filepath.endswith(".pickle")

    def pad_collate(batch):
        B = len(batch)
        if program_mode == "contrastive":
            X1, X2 = zip(*batch)
            X = X1 + X2
        else:
            raise NotImplementedError()

        # Create padded tensor for batch, [B, T] or [2B, T]
        X = pad_sequence(X, batch_first=True, padding_value=pad_id)

        if program_mode == "contrastive":
            # Reshape X to [B, 2, T]
            T = X.size(-1)
            X = torch.reshape(X, (2, B, -1))
            X = torch.transpose(X, 0, 1)
            assert X.shape == (B, 2, T)
        return (X, None)

    train_dataset = PrecomputedDataset(
        train_filepath,
        min_alternatives=2,
        program_mode="contrastive",
        limit_size=limit_dataset_size,
        sp=sp,
        subword_regularization_alpha=subword_regularization_alpha,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=num_workers, drop_last=True,
    )

    # Create model
    if loss_mode == "infonce":
        model = CodeMoCo(sp.GetPieceSize(), pad_id=pad_id)
        logger.info(f"Created CodeMoCo model with {count_parameters(model)} params")
    elif loss_mode == "mlm":
        model = CodeMLM(sp.GetPieceSize(), pad_id=pad_id)
        logger.info(f"Created CodeMLM model with {count_parameters(model)} params")
    model = nn.DataParallel(model)
    model = model.cuda() if use_cuda else model
    params = model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=1e-9)

    global_step = 0
    min_eval_loss = float("inf")
    for epoch in tqdm.trange(1, num_epochs + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            if loss_mode == "infonce":
                imgs, _ = batch
                if use_cuda:
                    imgs = imgs.cuda()
                imgs_k, imgs_q = imgs[:, 0, :], imgs[:, 1, :]
                output, target = model(imgs_q, imgs_k)
                loss = F.cross_entropy(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                train_metrics = {
                    "loss": loss,
                    "log": {
                        "pretrain/loss": loss.item(),
                        "pretrain/acc@1": acc1[0].item(),
                        "pretrain/acc@5": acc5[0].item(),
                        "pretrain/queue_ptr": model.module.queue_ptr.item(),
                    },
                }
                loss = train_metrics["loss"]
            loss.backward()
            optimizer.step()

            # Log loss
            global_step += 1
            wandb.log(dict(epoch=epoch, **train_metrics["log"]), step=global_step)
            pbar.set_description(f"epoch {epoch} loss {loss.item():.4f}")

            # Save checkpoint
            if save_every and global_step % save_every == 0:
                checkpoint = {
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": config,
                }
                model_file = run_dir / f"ckpt_pretrain_ep{epoch:04d}_step{global_step:07d}.pth"
                logger.info(f"Saving checkpoint to {model_file}...")
                torch.save(checkpoint, str(model_file.resolve()))
                # wandb.save(model_file)
                logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(pretrain)
