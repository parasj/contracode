import os
import random

import fire
from loguru import logger
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb

from representjs import RUN_DIR
from data.jsonl_dataset import get_csnjs_dataset
from representjs.data.csn_js_loader import javascript_dataloader
from representjs.utils import count_parameters
from representjs.models.transformer import TransformerModel
from representjs.decode import ids_to_strs, beam_search_decode

# Default argument values
DATA_DIR = "data/codesearchnet_javascript"
# CSNJS_TRAIN_FILEPATH = os.path.join(DATA_DIR, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
CSNJS_TRAIN_FILEPATH = os.path.join(DATA_DIR, "javascript_train_supervised.jsonl.gz")
CSNJS_VALID_FILEPATH = os.path.join(DATA_DIR, "javascript_valid_0.jsonl.gz")
CSNJS_TEST_FILEPATH = os.path.join(DATA_DIR, "javascript_test_0.jsonl.gz")
SPM_UNIGRAM_FILEPATH = os.path.join(DATA_DIR, "csnjs_8k_9995p_unigram_url.model")


def _evaluate(model, loader, sp: spm.SentencePieceProcessor, use_cuda=True,
             num_to_print=8, beam_search_k=5, max_decode_len=20):
    model.eval()
    pad_id = sp.PieceToId("[PAD]")

    with torch.no_grad():
        # Decode a single batch by beam search for visualization
        X, Y = next(iter(loader))
        X, Y = X[:num_to_print], Y[:num_to_print]
        if use_cuda:
            X = X.cuda(0)
        pred, scores = beam_search_decode(model, X, sp, k=beam_search_k, max_decode_len=max_decode_len)
        for i in range(X.size(0)):
            logger.info(f"Eval X:   \t\t\t{ids_to_strs(X[i], sp)}")
            logger.info(f"Eval GT Y:\t\t\t{ids_to_strs(Y[i], sp)}")
            for b in range(scores.size(1)):
                logger.info(f"Eval beam (score={scores[i, b]:.3f}):\t{pred[i][b]}")

        # Compute average loss
        total_loss = 0
        num_examples = 0
        pbar = tqdm.tqdm(loader, desc=f"evalaute")
        for X, Y in pbar:
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()
            # NOTE: X and Y are [B, max_seq_len] tensors (batch first)
            logits = model(X, Y[:, :-1])
            loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id)

            # TODO: Compute Precision/Recall/F1 and BLEU

            total_loss += loss.item() * X.size(0)
            num_examples += X.size(0)
            avg_loss = total_loss / num_examples
            pbar.set_description(f"evaluate average loss {avg_loss:.4f}")

        return avg_loss


def train(
        run_name: str,

        # Data
        train_filepath: str = CSNJS_TRAIN_FILEPATH,
        eval_filepath: str = CSNJS_VALID_FILEPATH,
        spm_filepath: str = SPM_UNIGRAM_FILEPATH,
        program_mode="identity",
        eval_program_mode="identity",
        label_mode="identifier",
        num_workers=1,
        limit_dataset_size=-1,

        # Model
        n_decoder_layers=4,

        # Optimization
        train_decoder_only: bool=False,
        num_epochs: int = 100,
        save_every: int = 2,
        batch_size: int = 256,
        lr: float = 8e-4,

        # Loss
        subword_regularization_alpha: float = 0,

        # Computational
        use_cuda: bool = True,
        seed: int=0
):
    """Train model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = RUN_DIR / run_name
    run_dir.mkdir(exist_ok=True, parents=True)
    logger.add(str((run_dir / "train.log").resolve()))
    logger.info(f"Saving logs, model checkpoints to {run_dir}")
    config = locals()
    logger.info(f"Config: {config}")
    wandb.init(name=run_name, config=config, job_type="training", project="code-representation", entity="ml4code")

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"

    train_augmentations = [
        {"fn": "sample_lines", "line_length_pct": 0.5},
        {"fn": "insert_var_declaration", "prob": 0.5},
        {"fn": "rename_variable", "prob": 0.5}
    ]

    eval_augmentations = []

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")

    # Create training dataset and dataloader
    logger.info(f"Training data path {train_filepath}")
    train_dataset = get_csnjs_dataset(train_filepath, label_mode=label_mode, limit_size=limit_dataset_size)
    logger.info(f"Training dataset size: {len(train_dataset)}")
    train_loader = javascript_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        augmentations=train_augmentations, sp=sp, program_mode=program_mode,
        subword_regularization_alpha=subword_regularization_alpha)

    # Create eval dataset and dataloader
    logger.info(f"Eval data path {eval_filepath}")
    eval_dataset = get_csnjs_dataset(eval_filepath, label_mode=label_mode, limit_size=limit_dataset_size)
    logger.info(f"Eval dataset size: {len(train_dataset)}")
    eval_loader = javascript_dataloader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        augmentations=eval_augmentations, sp=sp, program_mode=eval_program_mode,
        subword_regularization_alpha=subword_regularization_alpha)

    model = TransformerModel(n_tokens=sp.GetPieceSize(), pad_id=sp.PieceToId("[PAD]"), n_decoder_layers=n_decoder_layers)
    # model = TransformerModel(ntoken=sp.GetPieceSize(), ninp=512, pad_id=sp.PieceToId("[PAD]"))
    logger.info(f"Created TransformerModel with {count_parameters(model)} params")
    model = nn.DataParallel(model)
    model = model.cuda() if use_cuda else model
    params = model.decoder.parameters() if train_decoder_only else model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     lr,
    #     epochs=num_epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.02,  # Warm up for 2% of the total training time
    # )

    global_step = 0
    min_eval_loss = float("inf")
    for epoch in tqdm.trange(1, num_epochs + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        if train_decoder_only:
            model.encoder.eval()
            model.decoder.train()
        else:
            model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for X, Y in pbar:
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()
            optimizer.zero_grad()
            # NOTE: X and Y are [B, max_seq_len] tensors (batch first)
            logits = model(X, Y[:, :-1])
            loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Log loss
            global_step += 1
            wandb.log({
                "epoch": epoch,
                f"label-{label_mode}/train_loss": loss.item(),
                # "lr": scheduler.get_lr()
            }, step=global_step)
            pbar.set_description(f"epoch {epoch} loss {loss.item():.4f}")

        # Evaluate
        logger.info(f"Evaluating model after epoch {epoch} ({global_step} steps)...")
        eval_loss = _evaluate(model, eval_loader, sp, use_cuda=use_cuda)
        logger.info(f"Evaluation loss after epoch {epoch} ({global_step} steps): {eval_loss:.4f}")
        wandb.log({
            f"label-{label_mode}/eval_loss": eval_loss
        }, step=global_step)

        # Save checkpoint
        if save_every and epoch % save_every == 0 or eval_loss < min_eval_loss:
            checkpoint = {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
                "eval_loss": eval_loss
            }
            if eval_loss < min_eval_loss:
                logger.info(f"New best evaluation loss: prev {min_eval_loss:.4f} > new {eval_loss:.4f}")
                min_eval_loss = eval_loss
                model_file = run_dir / f"ckpt_best.pth"
            else:
                model_file = run_dir / f"ckpt_ep{epoch:04d}.pth"
            logger.info(f"Saving checkpoint to {model_file}...")
            torch.save(checkpoint, str(model_file.resolve()))
            logger.info("Done.")


if __name__ == "__main__":
    fire.Fire({"train": train})
