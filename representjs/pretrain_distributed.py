import os
import random
import time

import fire
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
import tqdm
import wandb
from loguru import logger
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence

from representjs import RUN_DIR, CSNJS_DIR
from data import transforms
from data.precomputed_dataset import PrecomputedDataset
from data.jsonl_dataset import get_csnjs_dataset
from models.code_moco import CodeMoCo
from utils import accuracy, count_parameters

DEFAULT_CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
DEFAULT_SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


def training_step(model, batch, use_cuda=False):
    imgs, _ = batch
    if use_cuda:
        imgs = imgs.cuda(non_blocking=True)
    imgs_k, imgs_q = imgs[:, 0, :], imgs[:, 1, :]
    output, target = model(imgs_q, imgs_k)
    loss = F.cross_entropy(output, target)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    logs = {
        "pretrain/loss": loss.item(),
        "pretrain/acc@1": acc1[0].item(),
        "pretrain/acc@5": acc5[0].item(),
        "pretrain/queue_ptr": model.module.queue_ptr.item(),
    }
    return {"loss": loss, "log": logs}


def pretrain(
    run_name: str,
    # Data
    train_filepath: str = DEFAULT_CSNJS_TRAIN_FILEPATH,
    spm_filepath: str = DEFAULT_SPM_UNIGRAM_FILEPATH,
    num_workers=1,
    limit_dataset_size=-1,
    # max_sequence_length=1024,
    subword_regularization_alpha: float = 0,
    program_mode="contrastive",
    min_alternatives=1,
    # Optimization
    num_epochs: int = 100,
    save_every: int = 1,
    batch_size: int = 256,
    lr: float = 8e-4,
    adam_betas=(0.9, 0.98),
    # Distributed
    rank: int = -1,
    dist_url: str = "env://",
    dist_backend: str = "nccl",
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

    assert (
        not use_cuda or torch.cuda.is_available()
    ), "CUDA not available. Check env configuration, or pass --use_cuda False"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = RUN_DIR / "{}_{}".format(run_name, int(time.time()))
    run_dir.mkdir(exist_ok=True, parents=True)
    config["run_dir"] = str(run_dir.resolve())
    logger.add(str((run_dir / "train.log").resolve()))
    logger.info(f"Saving logs, model checkpoints to {run_dir}")

    # Create training dataset and dataloader
    # assert train_filepath.endswith(".pickle")

    # Setup distributed
    ngpus_per_node = torch.cuda.device_count()
    config["world_size"] = ngpus_per_node  # only support 1 node
    mp.spawn(pretrain_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))


def pretrain_worker(gpu, ngpus_per_node, config):
    if config["rank"] % ngpus_per_node == 0:
        # NOTE: Should this be called by the 0th spawned process?
        wandb.init(
            name=config["run_name"], config=config, job_type="training", project="moco-pretrain", entity="ml4code",
        )

    if gpu is not None:
        logger.info("Use GPU: {} for training".format(gpu))

    if config["dist_url"] == "env://" and config["rank"] == -1:
        config["rank"] = int(os.environ["RANK"])
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    config["rank"] = config["rank"] * ngpus_per_node + gpu
    dist.init_process_group(
        backend=config["dist_backend"],
        init_method=config["dist_url"],
        world_size=config["world_size"],
        rank=config["rank"],
    )

    sp = spm.SentencePieceProcessor()
    sp.Load(config["spm_filepath"])
    pad_id = sp.PieceToId("[PAD]")

    def pad_collate(batch):
        B = len(batch)
        if config["program_mode"] == "contrastive":
            X1, X2 = zip(*batch)
            X = X1 + X2
        else:
            raise NotImplementedError()

        # Create padded tensor for batch, [B, T] or [2B, T]
        X = pad_sequence(X, batch_first=True, padding_value=pad_id)

        if config["program_mode"] == "contrastive":
            # Reshape X to [B, 2, T]
            T = X.size(-1)
            X = torch.reshape(X, (2, B, -1))
            X = torch.transpose(X, 0, 1)
            assert X.shape == (B, 2, T)
        return (X, None)

    # Create model
    model = CodeMoCo(sp.GetPieceSize(), pad_id=pad_id)
    logger.info(f"Created CodeMoCo model with {count_parameters(model)} params")

    assert config["use_cuda"]
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        config["batch_size"] = int(config["batch_size"] / ngpus_per_node)
        config["num_workers"] = int((config["num_workers"] + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=config["adam_betas"], eps=1e-9)

    # Setup data
    train_dataset = PrecomputedDataset(
        config["train_filepath"],
        min_alternatives=config["min_alternatives"],
        program_mode="contrastive",
        limit_size=config["limit_dataset_size"],
        sp=sp,
        subword_regularization_alpha=config["subword_regularization_alpha"],
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=pad_collate,
        # num_workers=config["num_workers"],
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler,
    )

    # Train
    global_step = 0
    for epoch in tqdm.trange(1, config["num_epochs"] + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            train_metrics = training_step(model, batch, use_cuda=config["use_cuda"])
            loss = train_metrics["loss"]
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_description(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")

            if config["rank"] % ngpus_per_node == 0:
                # Log loss
                wandb.log(dict(epoch=epoch, **train_metrics["log"]), step=global_step)

                # Save checkpoint
                if config["save_every"] and global_step % config["save_every"] == 0:
                    checkpoint = {
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "config": config,
                    }
                    model_file = os.path.join(
                        config["run_dir"], f"ckpt_pretrain_ep{epoch:04d}_step{global_step:07d}.pth"
                    )
                    logger.info(f"Saving checkpoint to {model_file}...")
                    torch.save(checkpoint, model_file)
                    # wandb.save(model_file)
                    logger.info("Done.")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    fire.Fire(pretrain)
