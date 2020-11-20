from pathlib import Path
import os
import random

import fire
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
# from bisect import bisect_right

import torch.nn.functional as F
import tqdm
import wandb
from loguru import logger

from data.util import Timer, normalize_program, EncodeAsIds
from models.typetransformer import TypeTransformer
from representjs import RUN_DIR
from utils import count_parameters, get_linear_schedule_with_warmup


# def find_gt(a, x):
#     'Find leftmost value greater than x'
#     i = bisect_right(a, x)
#     if i != len(a):
#         return a[i]
#     raise ValueError


class CloneProgramsDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, sp, subword_regularization_alpha=0., max_length=1024):
        self.sp = sp  # subword tokenizer
        self.subword_regularization_alpha = subword_regularization_alpha
        self.max_length = max_length
        self.bos_id = sp.PieceToId("<s>")
        self.eos_id = sp.PieceToId("</s>")

        self.all_solutions = []
        self.set_sizes = []
        # TODO: actually load real data, with filepath

        # Generate artificial data
        fake_num_problems = 20
        fake_num_solutions = 10
        # from random_word import RandomWords
        # r = RandomWords()
        import string
        def get_random_string(length):
            letters = string.ascii_lowercase
            result_str = ''.join(random.choice(letters) for i in range(length))
            return result_str
        for i in range(fake_num_problems):
            solutions = set()
            for j in range(fake_num_solutions):
                # solution = " ".join(r.get_random_words(minLength=10, maxLength=max_length))
                solution = get_random_string(100)
                solutions.add(solution)
            self.all_solutions.extend(solutions)
            self.set_sizes.append(len(solutions))

        self.cumulative_sizes = np.cumsum(self.set_sizes)
        self.num_problems = len(self.set_sizes)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        program = self.all_solutions[idx]
        return self.encode(program)
        # problem_idx = find_gt(self.cumulative_sizes, idx)

    def encode(self, program):
        program = normalize_program(program)
        program = EncodeAsIds(self.sp, self.subword_regularization_alpha, program)
        return torch.LongTensor([self.bos_id] + program[: (self.max_length - 2)] + [self.eos_id])


class ClonePairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: CloneProgramsDataset, max_pairs=-1, seed=None):
        self.dataset = dataset
        self.label = 0
        self.indices = self.get_indices()
        if max_pairs > 0:
            if seed is not None:
                np.random.seed(seed)
            subsample_idx = np.random.choice(len(self.indices), size=max_pairs)
            self.indices = self.indices[subsample_idx, :]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx1, idx2 = self.indices[idx]
        prog1_toks, prog2_toks = self.dataset[idx1], self.dataset[idx2]
        return prog1_toks, prog2_toks, self.label


class ClonePositivesDataset(ClonePairDataset):
    def get_indices(self):
        self.label = 1

        # Get all pairs of indices of programs solving the same problem
        indices = []
        for i in range(self.dataset.num_problems):
            # set_size = self.dataset.set_sizes[i]
            start_idx = self.dataset.cumulative_sizes[i-1] if i > 0 else 0
            end_idx = self.dataset.cumulative_sizes[i]  # exclusive, index of solution after the last solution for this problem
            for idx1 in range(start_idx, end_idx - 1):
                for idx2 in range(start_idx + 1, end_idx):
                    indices.append((idx1, idx2))
        logger.info(f"ClonePositvesDataset: Found {len(indices)} cloned pairs")
        return np.array(indices)


class CloneNegativesDataset(ClonePairDataset):
    def get_indices(self):
        # Get all pairs of indices of programs solving different problems
        indices = []
        for i in range(self.dataset.num_problems - 1):
            start_idx_i = self.dataset.cumulative_sizes[i-1] if i > 0 else 0
            end_idx_i = self.dataset.cumulative_sizes[i]

            for j in range(1, self.dataset.num_problems):
                start_idx_j = self.dataset.cumulative_sizes[j-1]
                end_idx_j = self.dataset.cumulative_sizes[j]

                for idx_i in range(start_idx_i, end_idx_i):
                    for idx_j in range(start_idx_j, end_idx_j):
                        indices.append((idx_i, idx_j))
        logger.info(f"CloneNegativesDataset: Found {len(indices)} non-cloned pairs")
        return np.array(indices)


def get_pad_collate(pad_id):
    def pad_collate(batch):
        X1, X2, labels = zip(*batch)
        X1 = pad_sequence(X1, batch_first=True, padding_value=pad_id)  # [B, T]
        X2 = pad_sequence(X2, batch_first=True, padding_value=pad_id)  # [B, T]
        labels = torch.tensor(labels, dtype=torch.long)
        return X1, X2, labels
    return pad_collate


def accuracy(output, target, topk=(1,), ignore_idx=[]):
    with torch.no_grad():
        maxk = max(topk)

        # Get top predictions per position that are not in ignore_idx
        target_vocab_size = output.size(2)
        keep_idx = torch.tensor([i for i in range(target_vocab_size) if i not in ignore_idx], device=output.device).long()
        _, pred = output[:, :, keep_idx].topk(maxk, 2, True, True)  # BxLx5
        pred = keep_idx[pred]  # BxLx5

        # Compute statistics over positions not labeled with an ignored idx
        correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).long()
        mask = torch.ones_like(target).long()
        for idx in ignore_idx:
            mask = mask.long() & (~target.eq(idx)).long()
        mask = mask.long()
        deno = mask.sum().item()
        correct = correct * mask.unsqueeze(-1)
        res = []
        for k in topk:
            correct_k = correct[..., :k].view(-1).float().sum(0)
            res.append(correct_k.item())

        return res, deno


def _evaluate(model, loader, sp: spm.SentencePieceProcessor, target_to_id, use_cuda=True, no_output_attention=False):
    model.eval()
    no_type_id = target_to_id["O"]
    any_id = target_to_id["$any$"]

    with torch.no_grad():
        # Accumulate metrics across batches to compute label-wise accuracy
        num1, num5, num_labels_total = 0, 0, 0
        num1_any, num5_any, num_labels_any_total = 0, 0, 0

        with Timer() as t:
            # Compute average loss
            total_loss = 0
            num_examples = 0
            pbar = tqdm.tqdm(loader, desc="evalaute")
            for X, lengths, output_attn, labels in pbar:
                if use_cuda:
                    X, lengths, output_attn, labels = X.cuda(), lengths.cuda(), output_attn.cuda(), labels.cuda()
                if no_output_attention:
                    logits = model(X, lengths, None)  # BxLxVocab
                else:
                    logits = model(X, lengths, output_attn)  # BxLxVocab
                # Compute loss
                loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=no_type_id)

                total_loss += loss.item() * X.size(0)
                num_examples += X.size(0)
                avg_loss = total_loss / num_examples

                # Compute accuracy
                (corr1_any, corr5_any), num_labels_any = accuracy(logits.cpu(), labels.cpu(), topk=(1, 5), ignore_idx=(no_type_id,))
                num1_any += corr1_any
                num5_any += corr5_any
                num_labels_any_total += num_labels_any

                (corr1, corr5), num_labels = accuracy(logits.cpu(), labels.cpu(), topk=(1, 5), ignore_idx=(no_type_id, any_id))
                num1 += corr1
                num5 += corr5
                num_labels_total += num_labels

                pbar.set_description(
                    f"evaluate average loss {avg_loss:.4f} num1 {num1_any} num_labels_any_total {num_labels_any_total} avg acc1_any {num1_any / (num_labels_any_total + 1e-6) * 100:.4f}"
                )

        # Average accuracies
        acc1 = float(num1) / num_labels_total * 100
        acc5 = float(num5) / num_labels_total * 100
        acc1_any = float(num1_any) / num_labels_any_total * 100
        acc5_any = float(num5_any) / num_labels_any_total * 100

        logger.debug(f"Loss calculation took {t.interval:.3f}s")
        return (
            -acc1_any,
            {
                "eval/loss": avg_loss,
                "eval/acc@1": acc1,
                "eval/acc@5": acc5,
                "eval/num_labels": num_labels_total,
                "eval/acc@1_any": acc1_any,
                "eval/acc@5_any": acc5_any,
                "eval/num_labels_any": num_labels_any_total,
            },
        )


def train(
    run_name: str,
    # Data
    train_filepath: str,
    eval_filepath: str,
    spm_filepath: str,
    num_workers=1,
    max_seq_len=1024,
    max_eval_seq_len=1024,
    run_dir=RUN_DIR,
    balance_negatives=False,
    # max_positive_pairs=-1,
    # max_negative_pairs=-1,
    # Model
    resume_path: str = "",
    pretrain_resume_path: str = "",
    pretrain_resume_encoder_name: str = "encoder_q",  # encoder_q, encoder_k, encoder
    pretrain_resume_project: bool = False,
    encoder_type: str = "transformer",
    n_encoder_layers: int = 6,
    d_model: int = 512,
    # Optimization
    train_decoder_only: bool = False,
    num_epochs: int = 100,
    save_every: int = 10,
    batch_size: int = 256,
    lr: float = 8e-4,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.98,
    adam_eps: float = 1e-6,
    weight_decay: float = 0,
    warmup_steps: int = 5000,
    num_steps: int = 200000,
    # Augmentations
    subword_regularization_alpha: float = 0,
    # Computational
    use_cuda: bool = True,
    seed: int = 1,
):
    """Train model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if run_dir != RUN_DIR:
        run_dir = Path(run_dir)
    run_dir = run_dir / run_name
    run_dir.mkdir(exist_ok=True, parents=True)
    logger.add(str((run_dir / "train.log").resolve()))
    logger.info(f"Saving logs, model checkpoints to {run_dir}")
    config = locals()
    logger.info(f"Config: {config}")
    wandb.init(name=run_name, config=config, job_type="training", project="clone_detection", entity="ml4code")

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")
    pad_collate = get_pad_collate(pad_id)

    # Create training dataset and dataloader
    logger.info(f"Training data path {train_filepath}")
    train_programs = CloneProgramsDataset(train_filepath, sp, subword_regularization_alpha)
    train_positives = ClonePositivesDataset(train_programs)
    if balance_negatives:
        train_negatives = CloneNegativesDataset(train_programs, len(train_positives), seed)
    else:
        train_negatives = CloneNegativesDataset(train_programs)
    train_dataset = torch.utils.data.ConcatDataset([train_positives, train_negatives])
    logger.info(f"Training dataset size: {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=pad_collate
    )

    # Create eval dataset and dataloader
    logger.info(f"Eval data path {eval_filepath}")
    eval_programs = CloneProgramsDataset(eval_filepath, sp, subword_regularization_alpha)
    eval_positives = ClonePositivesDataset(eval_programs)
    eval_negatives = CloneNegativesDataset(eval_programs)
    eval_dataset = torch.utils.data.ConcatDataset([eval_positives, eval_negatives])
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_collate
    )

    # Create model
    model = CloneModel(
        n_tokens=sp.GetPieceSize(),
        pad_id=pad_id,
        encoder_type=encoder_type,
        n_encoder_layers=n_encoder_layers,
        d_model=d_model,
        n_hidden_output=n_hidden_output,
    )
    logger.info(f"Created CloneModel {encoder_type} with {count_parameters(model)} params")

    # Load pretrained checkpoint
    if pretrain_resume_path:
        assert not resume_path
        logger.info(
            f"Resuming training from pretraining checkpoint {pretrain_resume_path}, pretrain_resume_encoder_name={pretrain_resume_encoder_name}"
        )
        checkpoint = torch.load(pretrain_resume_path)
        pretrained_state_dict = checkpoint["model_state_dict"]
        encoder_state_dict = {}
        output_state_dict = {}
        assert pretrain_resume_encoder_name in ["encoder_k", "encoder_q", "encoder"]

        for key, value in pretrained_state_dict.items():
            if key.startswith(pretrain_resume_encoder_name + ".") and "project_layer" not in key:
                remapped_key = key[len(pretrain_resume_encoder_name + ".") :]
                logger.debug(f"Remapping checkpoint key {key} to {remapped_key}. Value mean: {value.mean().item()}")
                encoder_state_dict[remapped_key] = value
            # if key.startswith(pretrain_resume_encoder_name + ".") and "project_layer.0." in key and pretrain_resume_project:
            #     remapped_key = key[len(pretrain_resume_encoder_name + ".project_layer.") :]
            #     logger.debug(f"Remapping checkpoint project key {key} to output key {remapped_key}. Value mean: {value.mean().item()}")
            #     output_state_dict[remapped_key] = value
        model.encoder.load_state_dict(encoder_state_dict)
        # TODO: check for head key rather than output for MLM
        model.output.load_state_dict(output_state_dict, strict=False)
        logger.info(f"Loaded state dict from {pretrain_resume_path}")

    # Set up optimizer
    model = nn.DataParallel(model)
    model = model.cuda() if use_cuda else model
    wandb.watch(model, log="all")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_steps)
    epoch = 0
    global_step = 0
    min_eval_metric = float("inf")

    if resume_path:
        assert not pretrain_resume_path
        logger.info(f"Resuming training from checkpoint {resume_path}")
        checkpoint = torch.load(resume_path)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        min_eval_metric = checkpoint["min_eval_metric"]

    # Eval metric history
    max_eval_metrics = {}

    # Evaluate initial metrics
    logger.info(f"Evaluating model after epoch {epoch} ({global_step} steps)...")
    eval_metric, eval_metrics = _evaluate(
        model, eval_loader, sp, target_to_id=target_to_id, use_cuda=use_cuda, no_output_attention=no_output_attention
    )
    for metric, value in eval_metrics.items():
        logger.info(f"Evaluation {metric} after epoch {epoch} ({global_step} steps): {value:.4f}")
        max_eval_metrics[metric] = value
    eval_metrics["epoch"] = epoch
    wandb.log(eval_metrics, step=global_step)
    wandb.log({k + "_max": v for k, v in max_eval_metrics.items()}, step=global_step)

    # TODO: Update below to handle single logit

    for epoch in tqdm.trange(epoch + 1, num_epochs + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for X1, X2, lengths1, lengths2, labels in pbar:
            if use_cuda:
                X1, X2, lengths1, lengths2, labels = X1.cuda(), X2.cuda(), lengths1.cuda(), lengths2.cuda(), labels.cuda()
            optimizer.zero_grad()
            logits = model(X1, X2, lengths1, lengths2)  # BxLxVocab
            # loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=no_type_id)
            # TODO: logistic loss!
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Compute accuracy in training batch
            (corr1_any, corr5_any), num_labels_any = accuracy(logits, labels, topk=(1, 5), ignore_idx=(no_type_id,))
            acc1_any, acc5_any = corr1_any / num_labels_any * 100, corr5_any / num_labels_any * 100
            (corr1, corr5), num_labels = accuracy(logits, labels, topk=(1, 5), ignore_idx=(no_type_id, any_id))
            acc1, acc5 = corr1 / num_labels * 100, corr5 / num_labels * 100

            # Log loss
            global_step += 1
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "train/acc@1": acc1,
                    "train/acc@5": acc5,
                    "train/acc@1_any": acc1_any,
                    "train/acc@5_any": acc5_any,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )
            pbar.set_description(f"epoch {epoch} loss {loss.item():.4f}")

        # Evaluate
        logger.info(f"Evaluating model after epoch {epoch} ({global_step} steps)...")
        eval_metric, eval_metrics = _evaluate(
            model, eval_loader, sp, target_to_id=target_to_id, use_cuda=use_cuda, no_output_attention=no_output_attention
        )
        for metric, value in eval_metrics.items():
            logger.info(f"Evaluation {metric} after epoch {epoch} ({global_step} steps): {value:.4f}")
            max_eval_metrics[metric] = max(value, max_eval_metrics[metric])
        eval_metrics["epoch"] = epoch
        wandb.log(eval_metrics, step=global_step)
        wandb.log({k + "_max": v for k, v in max_eval_metrics.items()}, step=global_step)

        # Save checkpoint
        if save_every and epoch % save_every == 0 or eval_metric < min_eval_metric:
            checkpoint = {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
                "eval_metric": eval_metric,
                "min_eval_metric": min_eval_metric,
            }
            if eval_metric < min_eval_metric:
                logger.info(f"New best evaluation metric: prev {min_eval_metric:.4f} > new {eval_metric:.4f}")
                min_eval_metric = eval_metric
                model_file = run_dir / "ckpt_best.pth"
            else:
                model_file = run_dir / f"ckpt_ep{epoch:04d}.pth"
            logger.info(f"Saving checkpoint to {model_file}...")
            torch.save(checkpoint, str(model_file.resolve()))
            logger.info("Done.")


# def eval(
#     # Data
#     eval_filepath: str,
#     type_vocab_filepath: str,
#     spm_filepath: str,
#     num_workers=1,
#     max_seq_len=-1,
#     # Model
#     resume_path: str = "",
#     no_output_attention: bool = False,
#     encoder_type: str = "transformer",
#     n_encoder_layers: int = 6,
#     d_model: int = 512,
#     # Output layer hparams
#     d_out_projection: int = 512,
#     n_hidden_output: int = 1,
#     # Optimization
#     batch_size=16,
#     # Loss
#     subword_regularization_alpha: float = 0,
#     # Computational
#     use_cuda: bool = True,
#     seed: int = 0,
# ):
#     """Evaluate model"""
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)

#     config = locals()
#     logger.info(f"Config: {config}")

#     if use_cuda:
#         assert torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"

#     sp = spm.SentencePieceProcessor()
#     sp.Load(spm_filepath)
#     pad_id = sp.PieceToId("[PAD]")

#     # Create eval dataset and dataloader
#     logger.info(f"Eval data path {eval_filepath}")
#     eval_dataset = # TODO
#     logger.info(f"Eval dataset size: {len(eval_dataset)}")
#     eval_loader = torch.utils.data.DataLoader(
#         eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
#     )

#     # Create model
#     model = TypeTransformer(
#         n_tokens=sp.GetPieceSize(),
#         n_output_tokens=len(id_to_target),
#         pad_id=pad_id,
#         encoder_type=encoder_type,
#         n_encoder_layers=n_encoder_layers,
#         d_model=d_model,
#         d_out_projection=d_out_projection,
#         n_hidden_output=n_hidden_output,
#     )
#     logger.info(f"Created TypeTransformer {encoder_type} with {count_parameters(model)} params")
#     model = nn.DataParallel(model)
#     model = model.cuda() if use_cuda else model

#     model.eval()
#     with torch.no_grad():
#         # Load checkpoint
#         logger.info(f"Loading parameters from {resume_path}")
#         checkpoint = torch.load(resume_path)
#         model.module.load_state_dict(checkpoint["model_state_dict"])
#         epoch = checkpoint["epoch"]
#         global_step = checkpoint["global_step"]

#         # Evaluate metrics
#         logger.info(f"Evaluating model after epoch {epoch} ({global_step} steps)...")
#         _, eval_metrics = _evaluate(
#             model, eval_loader, sp, target_to_id=target_to_id, use_cuda=use_cuda, no_output_attention=no_output_attention
#         )
#         for metric, value in eval_metrics.items():
#             logger.info(f"Evaluation {metric} after epoch {epoch} ({global_step} steps): {value:.4f}")



if __name__ == "__main__":
    fire.Fire({"train": train, "eval": eval})
