import os
import random

import fire
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import tqdm
import wandb
from loguru import logger

from data.jsonl_dataset import get_csnjs_dataset
from data.old_dataloader import javascript_dataloader
from data.util import Timer
from metrics.f1 import F1MetricMethodName
from models.transformer import TransformerModel, Seq2SeqLSTM
from representjs import RUN_DIR
from utils import count_parameters, get_linear_schedule_with_warmup
from decode import ids_to_strs, beam_search_decode, greedy_decode

# Default argument values
DATA_DIR = "data/codesearchnet_javascript"
# CSNJS_TRAIN_FILEPATH = os.path.join(DATA_DIR, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
CSNJS_TRAIN_FILEPATH = os.path.join(DATA_DIR, "javascript_train_supervised.jsonl.gz")
CSNJS_VALID_FILEPATH = os.path.join(DATA_DIR, "javascript_valid_0.jsonl.gz")
CSNJS_TEST_FILEPATH = os.path.join(DATA_DIR, "javascript_test_0.jsonl.gz")
SPM_UNIGRAM_FILEPATH = os.path.join(DATA_DIR, "csnjs_8k_9995p_unigram_url.model")


def _evaluate(
    model, loader, sp: spm.SentencePieceProcessor, use_cuda=True, num_to_print=8, beam_search_k=5, max_decode_len=20, loss_type="nll_token",
):
    model.eval()
    pad_id = sp.PieceToId("[PAD]")

    with torch.no_grad():
        # with Timer() as t:
        #     # Decode a single batch by beam search for visualization
        #     X, Y, X_lengths, _ = next(iter(loader))
        #     X, Y = X[:num_to_print], Y[:num_to_print]
        #     if use_cuda:
        #         X = X.cuda()
        #         X_lengths = X.cuda()
        #     pred, scores = beam_search_decode(model, X, X_lengths, sp, k=beam_search_k, max_decode_len=max_decode_len)
        #     for i in range(X.size(0)):
        #         logger.info(f"Eval X:   \t\t\t{ids_to_strs(X[i], sp)}")
        #         logger.info(f"Eval GT Y:\t\t\t{ids_to_strs(Y[i], sp)}")
        #         for b in range(scores.size(1)):
        #             logger.info(f"Eval beam (score={scores[i, b]:.3f}):\t{pred[i][b]}")
        # logger.debug(f"Decode time for {num_to_print} samples took {t.interval:.3f}")

        with Timer() as t:
            # Compute average loss
            total_loss = 0
            num_examples = 0
            pbar = tqdm.tqdm(loader, desc="evalaute")
            for X, Y, X_lengths, Y_lengths in pbar:
                if use_cuda:
                    X, Y = X.cuda(), Y.cuda()
                    X_lengths, Y_lengths = X_lengths.cuda(), Y_lengths.cuda()
                # NOTE: X and Y are [B, max_seq_len] tensors (batch first)
                logits = model(X, Y[:, :-1], X_lengths, Y_lengths)
                if loss_type == "nll_sequence":
                    loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id, reduction="sum")
                    loss = loss / X.size(0)  # Average over num sequences, not target sequence lengths
                    # Thus, minimize bits per sequence.
                elif loss_type == "nll_token":
                    loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id,)

                # TODO: Compute Precision/Recall/F1 and BLEU

                total_loss += loss.item() * X.size(0)
                num_examples += X.size(0)
                avg_loss = total_loss / num_examples
                pbar.set_description(f"evaluate average loss {avg_loss:.4f}")
        logger.debug(f"Loss calculation took {t.interval:.3f}s")
        return avg_loss


def calculate_f1_metric(
    metric: F1MetricMethodName,
    model,
    test_loader,
    sp: spm.SentencePieceProcessor,
    use_cuda=True,
    use_beam_search=True,
    beam_search_k=10,
    per_node_k=None,
    max_decode_len=20,  # see empirical evaluation of CDF of subwork token lengths
    beam_search_sampler="deterministic",
    top_p_threshold=0.9,
    top_p_temperature=1.0,
    logger_fn=None,
    constrain_decoding=False,
):
    with Timer() as t:
        sample_generations = []
        n_examples = 0
        precision, recall, f1 = 0.0, 0.0, 0.0
        with tqdm.tqdm(test_loader, desc="test") as pbar:
            for X, Y, _, _ in pbar:
                if use_cuda:
                    X, Y = X.cuda(), Y.cuda()  # B, L
                with Timer() as t:
                    # pred, scores = beam_search_decode(model, X, sp, k=beam_search_k, max_decode_len=max_decode_len)
                    if use_beam_search:
                        pred, _ = beam_search_decode(
                            model,
                            X,
                            sp,
                            max_decode_len=max_decode_len,
                            constrain_decoding=constrain_decoding,
                            k=beam_search_k,
                            per_node_k=per_node_k,
                            sampler=beam_search_sampler,
                            top_p_threshold=top_p_threshold,
                            top_p_temperature=top_p_temperature,
                        )
                    else:
                        pred = greedy_decode(model, X, sp, max_decode_len=max_decode_len)
                for i in range(X.size(0)):
                    gt_identifier = ids_to_strs(Y[i], sp)
                    pred_dict = {"gt": gt_identifier}
                    if use_beam_search:
                        top_beam = pred[i][0]
                        tqdm.tqdm.write("{:>20} vs. gt {:<20}".format(pred[i][0], gt_identifier))
                        for i, beam_result in enumerate(pred[i]):
                            pred_dict[f"pred_{i}"] = beam_result
                    else:
                        top_beam = pred[i]
                        pred_dict[f"pred_{i}"] = top_beam
                    sample_generations.append(pred_dict)
                    precision_item, score_item, f1_item = metric(top_beam, gt_identifier)

                    B = X.size(0)
                    n_examples += B
                    precision += precision_item * B
                    precision_avg = precision / n_examples
                    recall += score_item * B
                    recall_avg = recall / n_examples
                    f1 += f1_item * B
                    if precision_avg or recall_avg:
                        f1_overall = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
                    else:
                        f1_overall = 0.0
                    item_metrics = {"precision_item": precision_item, "recall_item": score_item, "f1_item": f1_item}
                    avg_metrics = {
                        "precision_avg": precision_avg,
                        "recall_avg": recall_avg,
                        "f1_avg": f1 / n_examples,
                        "f1_overall": f1_overall,
                    }
                    pbar.set_postfix(avg_metrics)
                    if logger_fn is not None:
                        logger_fn(item_metrics)
                        logger_fn(avg_metrics)
    logger.debug(f"Test set evaluation (F1) took {t.interval:.3}s over {n_examples} samples")
    precision_avg = precision / n_examples
    recall_avg = recall / n_examples
    f1_overall = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
    return precision_avg, recall_avg, f1_overall, sample_generations


def calculate_nll(model, test_loader, sp: spm.SentencePieceProcessor, use_cuda=True, logger_fn=None):
    pad_id = sp.PieceToId("[PAD]")
    n_examples = 0
    test_nll = 0.0
    with tqdm.tqdm(test_loader, desc="Test (NLL)") as pbar:
        for X, Y, X_lengths, Y_lengths in pbar:
            B, L = X.shape
            if use_cuda:
                X, Y = X.cuda(), Y.cuda()  # B, L
                X_lengths, Y_lengths = X_lengths.cuda(), Y_lengths.cuda()
            pred_y = model(X, Y[:, :-1].to(X.device), X_lengths, Y_lengths)
            B, X, D = pred_y.shape
            loss = F.cross_entropy(pred_y.reshape(B * X, D), Y[:, 1:].reshape(B * X), ignore_index=pad_id, reduction="sum")

            n_examples += B
            test_nll += loss.item()
            metric_dict = {"test_nll": loss.item() / B, "test_nll_avg": test_nll / n_examples}
            if logger_fn is not None:
                logger_fn(metric_dict)
            pbar.set_postfix(metric_dict)
    return test_nll / n_examples


def test(
    checkpoint_file: str,
    test_filepath: str = CSNJS_TEST_FILEPATH,
    spm_filepath: str = SPM_UNIGRAM_FILEPATH,
    program_mode="identity",
    label_mode="identifier",
    num_workers=1,
    limit_dataset_size=-1,
    batch_size=8,
    model_type="transformer",
    n_decoder_layers=4,
    d_model=512,
    use_cuda: bool = True,
    beam_search_k=10,
    per_node_k=None,
    beam_search_sampler="deterministic",
    top_p_threshold=0.9,
    top_p_temperature=1.0,
):
    wandb.init(name=checkpoint_file, config=locals(), project="f1_eval", entity="ml4code")
    if use_cuda:
        assert torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)

    # Create test dataset and dataloader
    logger.info(f"Test data path {test_filepath}")
    test_dataset = get_csnjs_dataset(test_filepath, label_mode=label_mode, limit_size=limit_dataset_size)
    logger.info(f"Test dataset size: {len(test_filepath)}")
    test_loader = javascript_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        sp=sp,
        program_mode=program_mode,
        subword_regularization_alpha=0,
        augmentations=[],
    )

    pad_id = sp.PieceToId("[PAD]")
    if model_type == "transformer":
        model = TransformerModel(n_tokens=sp.GetPieceSize(), pad_id=pad_id, n_decoder_layers=n_decoder_layers, d_model=d_model)
        logger.info(f"Created TransformerModel with {count_parameters(model)} params")
    elif model_type == "lstm":
        model = Seq2SeqLSTM(n_tokens=sp.GetPieceSize(), pad_id=pad_id, d_model=d_model)
        logger.info(f"Created Seq2SeqLSTM with {count_parameters(model)} params")

    if use_cuda:
        logger.info("Using cuda")
        model = model.cuda()
    else:
        logger.info("Using CPU")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_file)
    pretrained_state_dict = checkpoint["model_state_dict"]
    print("CHECKPOINT", checkpoint_file)
    print("KEYS", checkpoint["model_state_dict"].keys())
    try:
        model.load_state_dict(pretrained_state_dict)
    except RuntimeError as e:
        logger.error(e)
        logger.error("Keys in checkpoint: " + str(list(pretrained_state_dict.keys())))
        raise e
    logger.info(f"Loaded state dict from {checkpoint_file}")

    # Make metric
    metric = F1MetricMethodName()
    model.eval()
    with torch.no_grad():
        precision, recall, f1, sample_generations = calculate_f1_metric(
            metric,
            model,
            test_loader,
            sp,
            use_cuda=use_cuda,
            logger_fn=wandb.log,
            beam_search_k=beam_search_k,
            per_node_k=per_node_k if (per_node_k is not None and per_node_k > 0) else None,
            beam_search_sampler=beam_search_sampler,
            top_p_threshold=top_p_threshold,
            top_p_temperature=top_p_temperature,
        )
    logger.info(f"Precision: {precision:.5f}%")
    logger.info(f"Recall: {recall:.5f}%")
    logger.info(f"F1: {f1:.5f}%")
    wandb.log(dict(precision_final=precision, recall_final=recall, f1_final=f1))

    # Evaluate NLL
    model.eval()
    with torch.no_grad():
        test_nll = calculate_nll(model, test_loader, sp, use_cuda=use_cuda, logger_fn=wandb.log)

    logger.info(f"NLL: {test_nll:.5f}")
    logger.info(f"Precision: {precision:.5f}%")
    logger.info(f"Recall: {recall:.5f}%")
    logger.info(f"F1: {f1:.5f}%")

    df_generations = pd.DataFrame(sample_generations)
    df_generations.to_pickle(os.path.join(wandb.run.dir, "sample_generations.pickle.gz"))
    wandb.save(os.path.join(wandb.run.dir, "sample_generations.pickle.gz"))


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
    model_type="transformer",
    n_decoder_layers=4,
    d_model: int = 512,
    resume_path: str = "",
    resume_encoder_name: str = "encoder_q",  # encoder_q, encoder_k, encoder
    resume_project: bool = False,
    # Optimization
    train_decoder_only: bool = False,
    num_epochs: int = 50,
    save_every: int = 2,
    batch_size: int = 256,
    lr: float = 8e-4,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.98,
    use_lr_warmup: bool = True,
    loss_type="nll_token",  # nll_token or nll_sequence
    # Loss
    subword_regularization_alpha: float = 0,
    # Computational
    use_cuda: bool = True,
    auto_test: bool = True,
    seed: int = 0,
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
    wandb.init(name=run_name, config=config, job_type="training", project="identifier-prediction", entity="ml4code")

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"

    train_augmentations = [
        {
            "fn": "sample_lines",
            "line_length_pct": 0.5,
        },  # WARN: this is a no-op because the arguments for sample_lines are prob and prob_keep_line
        # Also need to have options under an "options" key
        {"fn": "insert_var_declaration", "prob": 0.5},
        {"fn": "rename_variable", "prob": 0.5},
    ]
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")

    # Create training dataset and dataloader
    logger.info(f"Training data path {train_filepath}")
    train_dataset = get_csnjs_dataset(train_filepath, label_mode=label_mode, limit_size=limit_dataset_size)
    logger.info(f"Training dataset size: {len(train_dataset)}")
    train_loader = javascript_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        augmentations=train_augmentations,
        sp=sp,
        program_mode=program_mode,
        subword_regularization_alpha=subword_regularization_alpha,
    )

    # Create eval dataset and dataloader
    logger.info(f"Eval data path {eval_filepath}")
    eval_dataset = get_csnjs_dataset(eval_filepath, label_mode=label_mode, limit_size=limit_dataset_size)
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    eval_loader = javascript_dataloader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        augmentations=[],
        sp=sp,
        program_mode=eval_program_mode,
        subword_regularization_alpha=subword_regularization_alpha,
    )

    # Create model
    pad_id = sp.PieceToId("[PAD]")
    if model_type == "transformer":
        model = TransformerModel(n_tokens=sp.GetPieceSize(), pad_id=pad_id, n_decoder_layers=n_decoder_layers, d_model=d_model)
        logger.info(f"Created TransformerModel with {count_parameters(model)} params")
    elif model_type == "lstm":
        model = Seq2SeqLSTM(n_tokens=sp.GetPieceSize(), pad_id=pad_id, d_model=d_model)
        logger.info(f"Created Seq2SeqLSTM with {count_parameters(model)} params")

    # Set up optimizer
    model = nn.DataParallel(model)
    model = model.cuda() if use_cuda else model
    wandb.watch(model, log="all")
    params = model.module.decoder.parameters() if train_decoder_only else model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, betas=(adam_beta1, adam_beta2), eps=1e-9)
    if use_lr_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer, 5000, len(train_loader) * num_epochs)
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1.0)

    # Load checkpoint
    start_epoch = 1
    global_step = 0
    min_eval_loss = float("inf")
    if resume_path:
        logger.info(f"Resuming training from checkpoint {resume_path}, resume_encoder_name={resume_encoder_name}")
        checkpoint = torch.load(resume_path)
        assert resume_encoder_name in ["encoder_k", "encoder_q", "encoder", "supervised"]

        if resume_encoder_name == "supervised":
            # This checkpoint is the result of training with this script, not pretraining
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            min_eval_loss = checkpoint.get("min_eval_loss", checkpoint["eval_loss"])

            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]

            for _ in range(global_step):
                scheduler.step()
        else:
            pretrained_state_dict = checkpoint["model_state_dict"]
            encoder_state_dict = {}

            for key, value in pretrained_state_dict.items():
                if key.startswith(resume_encoder_name + ".") and "project_layer" not in key:
                    remapped_key = key[len(resume_encoder_name + ".") :]
                    logger.debug(f"Remapping checkpoint key {key} to {remapped_key}. Value mean: {value.mean().item()}")
                    encoder_state_dict[remapped_key] = value
                if key.startswith(resume_encoder_name + ".") and "project_layer.0." in key and resume_project:
                    remapped_key = key[len(resume_encoder_name + ".") :]
                    logger.debug(f"Remapping checkpoint project key {key} to {remapped_key}. Value mean: {value.mean().item()}")
                    encoder_state_dict[remapped_key] = value
            model.encoder.load_state_dict(encoder_state_dict, strict=False)
            logger.info(f"Loaded keys: {encoder_state_dict.keys()}")
        logger.info(f"Loaded state dict from {resume_path}")

    for epoch in tqdm.trange(start_epoch, num_epochs + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        if train_decoder_only:
            model.module.encoder.eval()
            model.module.decoder.train()
        else:
            model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for X, Y, X_lengths, Y_lengths in pbar:
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()
                X_lengths, Y_lengths = X_lengths.cuda(), Y_lengths.cuda()
            optimizer.zero_grad()
            # NOTE: X and Y are [B, max_seq_len] tensors (batch first)
            logits = model(X, Y[:, :-1], X_lengths, Y_lengths)
            if loss_type == "nll_sequence":
                loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id, reduction="sum")
                loss = loss / X.size(0)  # Average over num sequences, not target sequence lengths
                # Thus, minimize bits per sequence.
            elif loss_type == "nll_token":
                loss = F.cross_entropy(logits.transpose(1, 2), Y[:, 1:], ignore_index=pad_id,)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            global_step += 1
            wandb.log(
                {"epoch": epoch, f"label-{label_mode}/train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=global_step,
            )
            pbar.set_description(f"epoch {epoch} loss {loss.item():.4f}")

        # Evaluate
        logger.info(f"Evaluating model after epoch {epoch} ({global_step} steps)...")
        max_decode_len = 20 if label_mode == "identifier" else 200
        eval_loss = _evaluate(model, eval_loader, sp, use_cuda=use_cuda, max_decode_len=max_decode_len, loss_type=loss_type)
        logger.info(f"Evaluation loss after epoch {epoch} ({global_step} steps): {eval_loss:.4f}")
        wandb.log({"epoch": epoch, f"label-{label_mode}/eval_loss": eval_loss}, step=global_step)

        # Save checkpoint
        if save_every and epoch % save_every == 0 or eval_loss < min_eval_loss:
            if eval_loss < min_eval_loss:
                logger.info(f"New best evaluation loss: prev {min_eval_loss:.4f} > new {eval_loss:.4f}")
                min_eval_loss = eval_loss
                model_file = run_dir / "ckpt_best.pth"
            else:
                model_file = run_dir / f"ckpt_ep{epoch:04d}.pth"
            checkpoint = {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
                "eval_loss": eval_loss,
                "min_eval_loss": min_eval_loss,
            }
            logger.info(f"Saving checkpoint to {model_file}...")
            torch.save(checkpoint, str(model_file.resolve()))
            wandb.save(str(model_file.resolve()))
            logger.info("Done.")

    if auto_test:
        best_ckpt = run_dir / "ckpt_best.pth"
        test(
            str(best_ckpt.resolve()),
            CSNJS_TEST_FILEPATH,
            spm_filepath,
            program_mode,
            label_mode,
            num_workers,
            -1,
            n_decoder_layers=n_decoder_layers,
        )


if __name__ == "__main__":
    fire.Fire({"train": train, "test": test})
