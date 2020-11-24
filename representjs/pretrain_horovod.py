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
import horovod.torch as hvd

from models.code_mlm import CodeMLM, CodeContrastiveMLM
from representjs import RUN_DIR, CSNJS_DIR
from data.precomputed_dataset import PrecomputedDataset
from models.code_moco import CodeMoCo
from utils import accuracy, count_parameters, get_linear_schedule_with_warmup

DEFAULT_CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
DEFAULT_SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


def training_step(model, batch, use_cuda=False):
    imgs, lengths, _ = batch
    if use_cuda:
        imgs = imgs.cuda(non_blocking=True)
    imgs_k, imgs_q = imgs[:, 0, :], imgs[:, 1, :]
    lengths_k, lengths_q = lengths[:, 0], lengths[:, 1]
    output, target = model(imgs_q, imgs_k, lengths_k, lengths_q)
    loss = F.cross_entropy(output, target)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    logs = {
        "pretrain/loss": loss.item(),
        "pretrain/acc@1": acc1[0].item(),
        "pretrain/acc@5": acc5[0].item(),
        "pretrain/queue_ptr": model.queue_ptr.item(),
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


def training_step_mlm(sp, model, batch, mask_id: int, pad_id: int, vocab_start_idx: int, vocab_end_idx: int, use_cuda=True):
    seq, lengths, _ = batch  # B x L
    if use_cuda:
        seq = seq.cuda()
    B, L = seq.shape
    seq_masked, targets = mask_mlm(seq, pad_id, mask_id, vocab_start_idx, vocab_end_idx)
    # logger.debug(f"Example transform:\t{sp.DecodeIds(seq_masked[0].cpu().numpy().tolist())}")
    output = model(seq_masked, lengths)  # B x L x Vocab
    assert targets.shape == (B, L), f"{targets.shape} versus {B}x{L}"
    assert output.shape == (B, L, output.shape[-1]), output.shape
    loss = F.cross_entropy(output.flatten(end_dim=1), targets.flatten(), ignore_index=pad_id)
    acc1, acc5 = accuracy(output[targets != pad_id], targets[targets != pad_id], topk=(1, 5))
    return {
        "loss": loss,
        "log": {"pretrain/loss": loss.item(), "pretrain/acc@1": acc1[0].item(), "pretrain/acc@5": acc5[0].item()},
    }


def training_step_hybrid(sp, model, batch, mask_id, pad_id, vocab_start_idx, vocab_end_idx, use_cuda):
    imgs, _lengths, _ = batch
    # TODO: implement LSTM for hybrid model and pass lengths to model call
    imgs_k, imgs_q = imgs[:, 0, :], imgs[:, 1, :]
    imgs_q, mlm_targets = mask_mlm(imgs_q, pad_id, mask_id, vocab_start_idx, vocab_end_idx)
    if use_cuda:
        imgs_k = imgs_k.cuda(non_blocking=True)
        imgs_q = imgs_q.cuda(non_blocking=True)
        mlm_targets = mlm_targets.cuda(non_blocking=True)
    predicted_masked_tokens, moco_logits, moco_targets = model(imgs_k, imgs_q)
    moco_loss = F.cross_entropy(moco_logits, moco_targets)
    moco_acc1, moco_acc5 = accuracy(moco_logits, moco_targets, topk=(1, 5))
    mlm_loss = F.cross_entropy(predicted_masked_tokens.flatten(end_dim=1), mlm_targets.flatten(), ignore_index=pad_id)
    mlm_acc1, mlm_acc5 = accuracy(predicted_masked_tokens[mlm_targets != pad_id], mlm_targets[mlm_targets != pad_id], topk=(1, 5))
    loss = 4 * moco_loss + mlm_loss
    logs = {
        "pretrain/moco/loss": moco_loss.item(),
        "pretrain/moco/acc@1": moco_acc1[0].item(),
        "pretrain/moco/acc@5": moco_acc5[0].item(),
        "pretrain/moco/queue_ptr": model.queue_ptr.item(),
        "pretrain/mlm/loss": mlm_loss.item(),
        "pretrain/mlm/acc@1": mlm_acc1[0].item(),
        "pretrain/mlm/acc@5": mlm_acc5[0].item(),
        "pretrain/hybrid_loss": loss,
    }
    return {"loss": loss, "log": logs}


def pad_collate_contrastive(batch):
    B = len(batch)
    X1, X2 = zip(*batch)
    X = X1 + X2

    # Create tensor of sequence lengths, [B] or [2B]
    lengths = torch.tensor([len(x) for x in X], dtype=torch.long)

    # Create padded tensor for batch, [B, T] or [2B, T]
    X = pad_sequence(X, batch_first=True, padding_value=0)

    # Reshape X to [B, 2, T]
    T = X.size(-1)
    X = torch.reshape(X, (2, B, -1))
    X = torch.transpose(X, 0, 1)
    assert X.shape == (B, 2, T)
    lengths = torch.reshape(lengths, (2, B)).transpose(0, 1)
    assert lengths.shape == (B, 2)

    return X, lengths, None


def pad_collate(batch):
    B = len(batch)
    X = batch

    # Create tensor of sequence lengths, [B] or [2B]
    lengths = torch.tensor([len(x) for x in X], dtype=torch.long)

    # Create padded tensor for batch, [B, T] or [2B, T]
    X = pad_sequence(X, batch_first=True, padding_value=0)

    return X, lengths, None


def pretrain(
    run_name: str,
    #
    # Data
    train_filepath: str = DEFAULT_CSNJS_TRAIN_FILEPATH,
    spm_filepath: str = DEFAULT_SPM_UNIGRAM_FILEPATH,
    num_workers=1,
    limit_dataset_size=-1,
    max_length=1024,
    subword_regularization_alpha: float = 0,
    program_mode="contrastive",
    loss_mode="infonce",  # infonce, mlm, or hybrid
    min_alternatives=1,
    #
    # Model
    resume_path: str = "",
    encoder_type: str = "transformer",
    lstm_project_mode: str = "hidden",
    n_encoder_layers: int = 6,
    d_model: int = 512,
    n_head: int = 8,
    #
    # Optimization
    num_epochs: int = 100,
    save_every: int = 1,
    batch_size: int = 256,
    lr: float = 8e-4,
    weight_decay: float = 0,
    adam_betas=(0.9, 0.98),
    warmup_steps: int = 5000,
    num_steps: int = 600000,
    #
    # Horovod
    use_adasum: bool = False,
    fp16_allreduce: bool = False,
    gradient_predivide_factor: float = 1.0,
    #
    # Computational
    use_cuda: bool = True,
    seed: int = 0,
):
    hvd.init()

    logger.info("L:", n_encoder_layers, type(n_encoder_layers))
    logger.info("H:", d_model, type(d_model))
    logger.info("A:", n_head, type(n_head))
    run_name = str(run_name)  # support numerical run ids
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_job_hostname = os.environ.get("SLURM_JOB_NODELIST")
    config = locals()
    logger.info(f"Config = \n{config}")
    logger.info("Training configuration: {}".format(config))
    logger.info(f"CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
    logger.info(f"CUDA_DEVICE_ORDER = '{os.environ.get('CUDA_DEVICE_ORDER')}'")

    assert program_mode in ["contrastive", "identity", "augmentation"]
    assert loss_mode == "infonce" or loss_mode == "mlm" or loss_mode == "hybrid"
    assert not (program_mode == "contrastive" and loss_mode == "mlm")
    assert not (program_mode != "contrastive" and (loss_mode == "hybrid" or loss_mode == "infonce"))
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
    gpu = hvd.local_rank()
    ngpus_per_node = 1
    chief_node = gpu == 0
    assert gpu is not None

    if chief_node:
        if config["loss_mode"] == "mlm":
            project = "bert-pretrain"
        elif config["loss_mode"] == "infonce":
            project = "moco-pretrain"
        elif config["loss_mode"] == "hybrid":
            project = "hybrid"
        wandb.init(name=config["run_name"], config=config, job_type="training", project=project, entity="ml4code")

    logger.info("Use GPU: {} for training".format(gpu))
    torch.cuda.set_device(gpu)
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {"num_workers": 1, "pin_memory": True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (
        kwargs.get("num_workers", 0) > 0
        and hasattr(mp, "_supports_context")
        and mp._supports_context
        and "forkserver" in mp.get_all_start_methods()
    ):
        kwargs["multiprocessing_context"] = "forkserver"

    sp = spm.SentencePieceProcessor()
    sp.Load(config["spm_filepath"])
    pad_id = sp.PieceToId("[PAD]")
    logger.info("pad_id {}", pad_id)
    assert pad_id == 0  # hard coded in pad_collate
    mask_id = sp.PieceToId("[MASK]")

    # Create model
    if config["loss_mode"] == "infonce":
        # TODO(ajay): Support n_head argument, check how d_model is being used (why not in encoder config dict?)
        model = CodeMoCo(
            sp.GetPieceSize(),
            pad_id=pad_id,
            d_model=config["d_model"],
            encoder_config=dict(
                encoder_type=config["encoder_type"],
                lstm_project_mode=config["lstm_project_mode"],
                n_encoder_layers=config["n_encoder_layers"],
            ),
        )
        logger.info(f"Created CodeMoCo model with {count_parameters(model)} params")
    elif config["loss_mode"] == "mlm":
        model = CodeMLM(
            sp.GetPieceSize(),
            pad_id=pad_id,
            encoder_type=config["encoder_type"],
            n_encoder_layers=config["n_encoder_layers"],
            d_model=config["d_model"],
            n_head=config["n_head"],
            d_ff=4 * config["d_model"],
        )
        logger.info(f"Created CodeMLM model with {count_parameters(model)} params")
    elif config["loss_mode"] == "hybrid":
        model = CodeContrastiveMLM(
            sp.GetPieceSize(),
            pad_id=pad_id,
            n_encoder_layers=config["n_encoder_layers"],
            d_model=config["d_model"],
            n_head=config["n_head"],
            d_ff=4 * config["d_model"],
            use_horovod=True,
        )
        logger.info(f"Created CodeContrastiveMLM model with {count_parameters(model)} params")
    else:
        raise ValueError(f"Bad loss mode {config['loss_mode']}")

    assert config["use_cuda"]
    model.cuda()
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    # config["batch_size"] = int(config["batch_size"] / ngpus_per_node)
    # config["num_workers"] = int((config["num_workers"] + ngpus_per_node - 1) / ngpus_per_node)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # define optimizer
    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not config["use_adasum"] else 1
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if config["use_adasum"] and hvd.nccl_built():
        lr_scaler = hvd.local_size()
    # Horovod: scale learning rate by lr_scaler.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"] * lr_scaler, betas=config["adam_betas"], eps=1e-6, weight_decay=config["weight_decay"]
    )
    sched = get_linear_schedule_with_warmup(optimizer, config["warmup_steps"], config["num_steps"])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if config["fp16_allreduce"] else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Adasum if config["use_adasum"] else hvd.Average,
        gradient_predivide_factor=config["gradient_predivide_factor"],
    )

    # Load checkpoint
    if config["resume_path"]:
        logger.info(f"Loading parameters from {config['resume_path']}")
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % hvd.rank()}
        checkpoint = torch.load(config["resume_path"], map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        start_global_step = checkpoint["global_step"]
    else:
        start_epoch = 1
        start_global_step = 0

    # Setup data
    train_dataset = PrecomputedDataset(
        config["train_filepath"],
        min_alternatives=config["min_alternatives"],
        program_mode=config["program_mode"],
        limit_size=config["limit_dataset_size"],
        sp=sp,
        subword_regularization_alpha=config["subword_regularization_alpha"],
        max_length=config["max_length"],
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=pad_collate_contrastive if config["program_mode"] == "contrastive" else pad_collate,
        drop_last=True,
        sampler=train_sampler,
        **kwargs,
    )

    # Train
    global_step = 0
    while global_step < start_global_step:
        sched.step()
        global_step += 1
    for epoch in tqdm.trange(start_epoch, config["num_epochs"] + 1, desc="training", unit="epoch", leave=False):
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
                    sp, model, batch, pad_id=pad_id, mask_id=mask_id, vocab_start_idx=8, vocab_end_idx=7999, use_cuda=config["use_cuda"]
                )
            elif config["loss_mode"] == "hybrid":
                train_metrics = training_step_hybrid(
                    sp, model, batch, mask_id=mask_id, pad_id=pad_id, vocab_start_idx=0, vocab_end_idx=7999, use_cuda=config["use_cuda"]
                )
            else:
                raise ValueError("Bad loss type")
            loss = train_metrics["loss"]
            loss.backward()
            optimizer.step()
            sched.step()

            global_step += 1
            pbar.set_description(f"epoch {epoch} gpu {gpu} step {global_step} loss {loss.item():.4f}")

            if chief_node:
                wandb.log(dict(lr=sched.get_last_lr()[0]))
                wandb.log(dict(epoch=epoch, **train_metrics["log"]), step=global_step)

                # Save checkpoint
                if config["save_every"] and global_step % config["save_every"] == 0:
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "config": config,
                    }
                    model_file = os.path.join(config["run_dir"], f"ckpt_pretrain_ep{epoch:04d}_step{global_step:07d}.pth")
                    logger.info(f"Saving checkpoint to {model_file}...")
                    torch.save(checkpoint, model_file)
                    wandb.save(str(model_file))
                    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(pretrain)
