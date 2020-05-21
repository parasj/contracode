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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence

from models.code_mlm import CodeMLM
from representjs import RUN_DIR, CSNJS_DIR
from data.precomputed_dataset import PrecomputedDataset
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


def mask_mlm(seq, pad_id, mask_id, vocab_start_range, vocab_end_range):
    # The training data generator chooses 15% of the token positions at random for prediction.
    # If the i-th token is chosen, we replace the i-th token with
    # (0) not masked
    # (1) the [MASK] token 80% of the time (0.12)
    # (2) a random token 10% of the time (0.015)
    # (3) the unchanged i-th token 10% of the time (0.015)
    #
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py#L63
    rand_replacements = torch.zeros_like(seq, dtype=torch.long).random_(vocab_start_range, vocab_end_range)

    masked_tokens = (torch.rand_like(seq, dtype=torch.float) < 0.15) & (seq != pad_id)
    mask_type_prob = torch.rand_like(seq, dtype=torch.float)
    mask_token_prob = (mask_type_prob < 0.8) & masked_tokens
    random_token_prob = (mask_type_prob < 0.9) & (mask_type_prob >= 0.8) & masked_tokens
    identity_token_prob = (mask_type_prob >= 0.9) & masked_tokens
    assert torch.sum(masked_tokens) == torch.sum(mask_token_prob | random_token_prob | identity_token_prob)

    targets = torch.zeros_like(seq).fill_(pad_id)
    targets[masked_tokens] = seq[masked_tokens]

    seq[mask_token_prob] = mask_id
    seq[random_token_prob] = rand_replacements[random_token_prob]
    return seq, targets


def training_step_mlm(model, batch, mask_id: int, pad_id: int, vocab_start_idx: int, vocab_end_idx: int, use_cuda=True):
    seq, _ = batch  # B x L
    if use_cuda:
        seq = seq.cuda()
    B, L = seq.shape
    seq_masked, targets = mask_mlm(seq, pad_id, mask_id, vocab_start_idx, vocab_end_idx)
    output = model(seq_masked)  # B x L x Vocab
    assert targets.shape == (B, L), f"{targets.shape} versus {B}x{L}"
    assert output.shape == (B, L, output.shape[-1]), output.shape
    loss = F.cross_entropy(output.flatten(end_dim=1), targets.flatten(), ignore_index=pad_id)
    acc1, acc5 = accuracy(output[targets != pad_id], targets[targets != pad_id], topk=(1, 5))
    return {
        "loss": loss,
        "log": {
            "pretrain/loss": loss.item(),
            "pretrain/acc@1": acc1[0].item(),
            "pretrain/acc@5": acc5[0].item(),
        },
    }


def pretrain(
    run_name: str,
    #
    # Data
    train_filepath: str = DEFAULT_CSNJS_TRAIN_FILEPATH,
    spm_filepath: str = DEFAULT_SPM_UNIGRAM_FILEPATH,
    num_workers=1,
    limit_dataset_size=-1,
    # max_sequence_length=1024,
    subword_regularization_alpha: float = 0,
    program_mode="contrastive",
    loss_mode="infonce",  # infonce or mlm
    min_alternatives=1,
    #
    # Optimization
    num_epochs: int = 100,
    save_every: int = 1,
    batch_size: int = 256,
    lr: float = 8e-4,
    adam_betas=(0.9, 0.98),
    #
    # Distributed
    rank: int = -1,
    dist_url: str = "env://",
    dist_backend: str = "nccl",
    #
    # Computational
    use_cuda: bool = True,
    seed: int = 0,
):
    run_name = str(run_name)  # support numerical run ids
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_job_hostname = os.environ.get("SLURM_JOB_NODELIST")
    config = locals()
    logger.info(f"Config = \n{config}")
    logger.info("Training configuration: {}".format(config))
    logger.info(f"CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
    logger.info(f"CUDA_DEVICE_ORDER = '{os.environ.get('CUDA_DEVICE_ORDER')}'")

    assert program_mode in ["contrastive", "identity", "augmentation"]
    assert loss_mode == "infonce" or loss_mode == "mlm"
    assert not (program_mode == "contrastive" and loss_mode == "mlm")
    assert not use_cuda or torch.cuda.is_available()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = RUN_DIR / "{}_{}".format(run_name, int(time.time()))
    run_dir.mkdir(exist_ok=True, parents=True)
    config["run_dir"] = str(run_dir.resolve())
    logger.add(str((run_dir / "train.log").resolve()))
    logger.info(f"Saving logs, model checkpoints to {run_dir}")

    # Create training dataset and dataloader
    assert train_filepath.endswith(".pickle") or train_filepath.endswith(".gz")

    # Setup distributed
    ngpus_per_node = torch.cuda.device_count()
    config["world_size"] = ngpus_per_node  # only support 1 node
    mp.spawn(pretrain_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))


def pretrain_worker(gpu, ngpus_per_node, config):
    if config["rank"] % ngpus_per_node == 0:
        # NOTE: Should this be called by the 0th spawned process?
        wandb.init(name=config["run_name"], config=config, job_type="training", project="moco-pretrain", entity="ml4code")

    if gpu is not None:
        logger.info("Use GPU: {} for training".format(gpu))

    if config["dist_url"] == "env://" and config["rank"] == -1:
        config["rank"] = int(os.environ["RANK"])
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    config["rank"] = config["rank"] * ngpus_per_node + gpu
    dist.init_process_group(
        backend=config["dist_backend"], init_method=config["dist_url"], world_size=config["world_size"], rank=config["rank"]
    )

    sp = spm.SentencePieceProcessor()
    sp.Load(config["spm_filepath"])
    pad_id = sp.PieceToId("[PAD]")
    mask_id = sp.PieceToId("[MASK]")

    def pad_collate(batch):
        B = len(batch)
        if config["program_mode"] == "contrastive":
            X1, X2 = zip(*batch)
            X = X1 + X2
        else:
            X = batch

        # Create padded tensor for batch, [B, T] or [2B, T]
        X = pad_sequence(X, batch_first=True, padding_value=pad_id)

        if config["program_mode"] == "contrastive":
            # Reshape X to [B, 2, T]
            T = X.size(-1)
            X = torch.reshape(X, (2, B, -1))
            X = torch.transpose(X, 0, 1)
            assert X.shape == (B, 2, T)
        return X, None

    # Create model
    if config["loss_mode"] == "infonce":
        model = CodeMoCo(sp.GetPieceSize(), pad_id=pad_id)
        logger.info(f"Created CodeMoCo model with {count_parameters(model)} params")
    elif config["loss_mode"] == "mlm":
        model = CodeMLM(sp.GetPieceSize(), pad_id=pad_id)
        logger.info(f"Created CodeMLM model with {count_parameters(model)} params")
    else:
        raise ValueError(f"Bad loss mode {config['loss_mode']}")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=config["adam_betas"], eps=1e-6, weight_decay=0.01)

    # Setup data
    train_dataset = PrecomputedDataset(
        config["train_filepath"],
        min_alternatives=config["min_alternatives"],
        program_mode=config["program_mode"],
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
            if config["loss_mode"] == "infonce":
                train_metrics = training_step(model, batch, use_cuda=config["use_cuda"])
            elif config["loss_mode"] == "mlm":
                # replace tokens randomly with tokens from _ (8)
                train_metrics = training_step_mlm(
                    model, batch, pad_id=pad_id, mask_id=mask_id, vocab_start_idx=8, vocab_end_idx=7999, use_cuda=config["use_cuda"]
                )
            else:
                raise ValueError("Bad loss type")
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
                    model_file = os.path.join(config["run_dir"], f"ckpt_pretrain_ep{epoch:04d}_step{global_step:07d}.pth")
                    logger.info(f"Saving checkpoint to {model_file}...")
                    torch.save(checkpoint, model_file)
                    # wandb.save(model_file)
                    logger.info("Done.")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    fire.Fire(pretrain)
