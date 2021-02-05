from collections import defaultdict
import gzip
import json
from pathlib import Path
import os
import random
import time

import fire
import numpy as np
import ray
import sentencepiece as spm
import textdistance
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

import torch.nn.functional as F
import tqdm
import wandb
from loguru import logger

from data.util import Timer, normalize_program, EncodeAsIds
from data.old_dataloader import _augment_server
from models.clone import CloneModel
from representjs import RUN_DIR
from utils import count_parameters, get_linear_schedule_with_warmup


class CloneProgramsDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, sp, subword_regularization_alpha=0.0, max_length=1024, augmentations=None):
        self.augmentations = augmentations
        self.sp = sp  # subword tokenizer
        self.subword_regularization_alpha = subword_regularization_alpha
        self.max_length = max_length
        self.bos_id = sp.PieceToId("<s>")
        self.eos_id = sp.PieceToId("</s>")

        self.all_solutions = []
        self.set_sizes = []
        # Load data
        data_file = gzip.open(filepath, "r") if filepath.endswith(".json.gz") else open(filepath, "r")
        data = json.load(data_file)
        data_file.close()
        # Flatten and compute solution set sizes
        for solutions in data:  # iterate, solutions consists of programs solving the same problem
            self.all_solutions.extend(solutions)
            self.set_sizes.append(len(solutions))

        self.cumulative_sizes = np.cumsum(self.set_sizes)
        self.num_problems = len(self.set_sizes)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx, augment=True):
        program = self.all_solutions[idx]
        if self.augmentations and augment:
            # Set up transformation input
            transform_payload = [dict(src=program, augmentations=self.augmentations)]
            program = _augment_server(transform_payload)
            assert isinstance(program, list) and len(program) == 1
            program = program[0]
        return self.encode(program)

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
        prog1_toks = self.dataset.__getitem__(idx1, augment=True)
        prog2_toks = self.dataset.__getitem__(idx2, augment=False)
        return prog1_toks, prog2_toks, self.label


class ClonePositivesDataset(ClonePairDataset):
    def get_indices(self):
        self.label = 1

        # Get all pairs of indices of programs solving the same problem
        indices = []
        for i in range(self.dataset.num_problems):
            # set_size = self.dataset.set_sizes[i]
            start_idx = self.dataset.cumulative_sizes[i - 1] if i > 0 else 0
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
            start_idx_i = self.dataset.cumulative_sizes[i - 1] if i > 0 else 0
            end_idx_i = self.dataset.cumulative_sizes[i]

            for j in range(1, self.dataset.num_problems):
                start_idx_j = self.dataset.cumulative_sizes[j - 1]
                end_idx_j = self.dataset.cumulative_sizes[j]

                for idx_i in range(start_idx_i, end_idx_i):
                    for idx_j in range(start_idx_j, end_idx_j):
                        indices.append((idx_i, idx_j))
        logger.info(f"CloneNegativesDataset: Found {len(indices)} non-cloned pairs")
        return np.array(indices)


def get_pad_collate(pad_id):
    def pad_collate(batch):
        X1, X2, labels = zip(*batch)
        X = X1 + X2

        # Create tensor of sequence lengths, [2B]
        lengths = torch.tensor([len(x) for x in X], dtype=torch.long)

        # Pad sequences
        X = pad_sequence(X, batch_first=True, padding_value=pad_id)  # [2B, T]

        # Create tensor of labels, [B]
        labels = torch.tensor(labels, dtype=torch.long)
        return X, lengths, labels

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


def best_binary_accuracy_score(y_true, y_score, num_thresholds=100):
    assert y_true.shape == y_score.shape
    thresholds = np.linspace(0, 1, num=num_thresholds)
    best_acc = 0.0
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype(y_true.dtype)
        acc = accuracy_score(y_true, y_pred)
        best_acc = max(best_acc, acc)
    return best_acc


def se_auc(auc, n_p, n_n):
    """Standard error of AUROC. Ported from https://rdrr.io/cran/auctestr/man/se_auc.html"""
    D_p = (n_p - 1) * ((auc/(2 - auc)) - auc**2)
    D_n = (n_n - 1) * ((2 * auc**2)/(1 + auc) - auc**2)
    SE_auc = np.sqrt((auc * (1 - auc) + D_p + D_n)/(n_p * n_n))
    return SE_auc


def _get_generator(data_loaders, pad_id, pbar=True, pbar_desc=""):
    """A generator zipping multiple data loaders"""
    if pbar:
        loader0 = tqdm.tqdm(data_loaders[0], desc=pbar_desc)
    else:
        loader0 = data_loaders[0]

    remaining_data_loaders = [iter(loader) for loader in data_loaders[1:]]

    for batch0 in loader0:
        batches = [batch0] + [next(loader) for loader in remaining_data_loaders]
        X, lengths, labels = zip(*batches)
        # In [11]: X[0].shape
        # Out[11]: torch.Size([32, 928])
        # In [12]: lengths[0].shape
        # Out[12]: torch.Size([32])
        # In [13]: labels[0].shape
        # Out[13]: torch.Size([16])

        # X is a tuple of [2B, L] dimensional tensors.
        # Rearrange and pad to get a single tensor.
        X_LB = [x.transpose(0, 1) for x in X]
        X_LAB = pad_sequence(X_LB, batch_first=False, padding_value=pad_id)
        X = X_LAB.transpose(0, 2)  # [2B, adversarial_samples, L]
        X = X.contiguous()

        lengths = torch.stack(lengths, dim=1)  # [2B, adversarial_samples]

        # just use labels = labels[0] since the labels don't
        # change between different augmentation samples
        labels = labels[0]  # [B]
        # labels = torch.stack(labels, dim=1)  # [B, adversarial_samples]

        yield X, lengths, labels


@torch.no_grad()
def _evaluate(model, loaders, sp: spm.SentencePieceProcessor, pad_id, use_cuda=True, save_path=None):
    model.eval()

    with Timer() as t:
        # Compute ROC AUC, AP
        num_examples_by_adversarial_samples = defaultdict(lambda: 0)
        y_true_by_adversarial_samples = defaultdict(list)
        y_scores_by_adversarial_samples = defaultdict(list)
        generator = _get_generator(loaders, pad_id=pad_id, pbar=True, pbar_desc="evaluate")

        def _summarize_stats(y_true, y_scores):
            ytc = np.concatenate(y_true)
            ysc = np.concatenate(y_scores)
            accuracy = best_binary_accuracy_score(ytc, ysc)
            num_pos, num_neg = np.sum(ytc), np.sum(ytc == 0)

            roc_auc, se_roc_auc, ap_score = 0, 0, 0
            if ytc.sum() != 0 and ytc.sum() < len(ytc):
                roc_auc = roc_auc_score(ytc, ysc)
                se_roc_auc = se_auc(roc_auc, num_pos, num_neg)
                ap_score = average_precision_score(ytc, ysc)

            return roc_auc, se_roc_auc, ap_score, accuracy, num_pos, num_neg

        for X, lengths, labels in generator:
            if use_cuda:
                X, lengths, labels = X.cuda(), lengths.cuda(), labels.cuda()
            two_B, adversarial_samples, L = X.shape

            # Compute similarity
            similarity = model(X.view(two_B*adversarial_samples, L), lengths.view(-1))  # B*adversarial_samples
            assert similarity.ndim == 1

            # Reduce similarity adversarially. Maximize over sliding windows so we can compute
            # metrics for 1, 2, ... adversarial_samples samples of transforms.
            similarity = similarity.view(two_B // 2, adversarial_samples)
            for window_size in range(1, adversarial_samples + 1):
                for start in range(0, adversarial_samples - window_size + 1):
                    window_similarity = similarity[:, start:start+window_size]
                    min_similarity, _ = torch.min(window_similarity, dim=1)
                    max_similarity, _ = torch.max(window_similarity, dim=1)
                    window_similarity = min_similarity * labels + max_similarity * (1 - labels)

                    num_examples_by_adversarial_samples[window_size] += 1
                    y_scores_by_adversarial_samples[window_size].append(window_similarity.cpu().numpy())
                    y_true_by_adversarial_samples[window_size].append(labels.cpu().numpy())

            if num_examples_by_adversarial_samples[adversarial_samples] % 100 == 0:
                for window_size in range(1, adversarial_samples + 1):
                    roc_auc, se_roc_auc, ap_score, accuracy, num_pos, num_neg = _summarize_stats(
                        y_true_by_adversarial_samples[window_size],
                        y_scores_by_adversarial_samples[window_size])
                    print(f"evaluate with {window_size} adversarial samples: roc_auc {roc_auc:.4f}±{se_roc_auc:.4f} (+{num_pos},-{num_neg}) ap {ap_score:.4f} acc {accuracy:.4f}")

    # Compute ROC AUC and AP
    metrics = {}

    for window_size in range(1, adversarial_samples + 1):
        roc_auc, se_roc_auc, ap_score, accuracy, num_pos, num_neg = _summarize_stats(
            y_true_by_adversarial_samples[window_size],
            y_scores_by_adversarial_samples[window_size])

        print(f"evaluate with {window_size} adversarial samples: roc_auc {roc_auc:.4f}±{se_roc_auc:.4f} (+{num_pos},-{num_neg}) ap {ap_score:.4f} acc {accuracy:.4f}")

        metrics.update({
            f"eval/roc_auc_score/adv{window_size}": roc_auc,
            f"eval/se_roc_auc/adv{window_size}": se_roc_auc,
            f"eval/ap_score/adv{window_size}": ap_score,
            f"eval/accuracy/adv{window_size}": accuracy,
            f"eval/num_examples/adv{window_size}": num_examples_by_adversarial_samples[window_size],
            f"eval/num_positive/adv{window_size}": num_pos,
            f"eval/num_negative/adv{window_size}": num_neg,
        })

    logger.debug(f"Loss calculation took {t.interval:.3f}s")

    if save_path:
        logger.info("Saving labels, scores and metrics to {}", save_path)
        torch.save({"y_true": dict(y_true_by_adversarial_samples), "y_scores": dict(y_scores_by_adversarial_samples),
                    "num_examples": dict(num_examples_by_adversarial_samples), "metrics": metrics}, save_path)

    return (-roc_auc, metrics)


@torch.no_grad()
def _levenshtein_similarity(a, b):
    distance = textdistance.levenshtein.distance(a, b)
    maxlen = max(len(a), len(b))
    ratio = (maxlen - distance) / float(maxlen)  # similarity ratio
    return ratio


@torch.no_grad()
def _evaluate_edit_distance(loaders, sp, pad_id, edit_distance_mode="tokens", save_path=None):
    # TODO: implement adversarial evaluation for edit distance
    # assert len(loaders) == 1
    # loader = loaders[0]

    assert edit_distance_mode == "tokens"
    ray.init()

    @ray.remote
    def _compute_similarity(X, lengths, labels):
        two_B, adversarial_samples, L = X.shape
        B = two_B // 2

        assert X.ndim == 3
        assert lengths.ndim == 2
        assert lengths.shape == (two_B, adversarial_samples)
        assert labels.shape == (B,)

        # Compute similarity
        X = X.view(2, B*adversarial_samples, L)
        lengths = lengths.view(2, B*adversarial_samples)
        similarity = torch.zeros(B*adversarial_samples, dtype=torch.float32)

        for i in range(B*adversarial_samples):
            a_len = lengths[0, i]
            b_len = lengths[1, i]
            a = list(X[0, i].numpy())[:a_len]  # remove padding
            b = list(X[1, i].numpy())[:b_len]  # remove padding
            # a = list(X[0, i].numpy())[1:a_len-1]  # remove bos_id, eos_id and padding
            # b = list(X[1, i].numpy())[1:b_len-1]  # remove bos_id, eos_id and padding
            similarity[i] = _levenshtein_similarity(a, b)  # B

        similarity = similarity.view(B, adversarial_samples)

        # Aggregate adversarially. similarity is [B, adversarial_samples]
        min_similarity, _ = torch.min(similarity, dim=1)
        max_similarity, _ = torch.max(similarity, dim=1)
        similarity = min_similarity * labels + max_similarity * (1 - labels)

        return similarity.numpy()  # [B,]

    with Timer() as t:
        # Compute average loss
        adversarial_samples = len(loaders)
        total_similarity = 0
        num_examples = 0

        y_true = []
        y_scores = []

        generator = _get_generator(loaders, pad_id=pad_id, pbar=True, pbar_desc="queue up evalaute")
        similarity_futures = []
        for X, lengths, labels in generator:
            f = _compute_similarity.remote(X, lengths, labels)
            similarity_futures.append(f)

            num_examples += X.size(1)
            y_true.append(labels.numpy())

        # Aggregate futures, compute ROC AUC, AP
        pbar = tqdm.tqdm(map(ray.get, similarity_futures), desc="get similarities")
        for i, similarity in enumerate(pbar):
            total_similarity += np.sum(similarity)
            avg_similarity = total_similarity / num_examples
            y_scores.append(similarity)

            if i % 10 == 0:
                ytc = np.concatenate(y_true[: i + 1])
                num_pos, num_neg = np.sum(ytc), np.sum(ytc == 0)
                if ytc.sum() != 0 and ytc.sum() < len(ytc):
                    ysc = np.concatenate(y_scores)
                    roc_auc = roc_auc_score(ytc, ysc)
                    se_roc_auc = se_auc(roc_auc, num_pos, num_neg)
                    ap_score = average_precision_score(ytc, ysc)
                else:
                    roc_auc = 0
                    ap_score = 0
                    se_roc_auc = 0
                pbar.set_description(f"evaluate with {adversarial_samples} adversarial samples: avg similarity {avg_similarity:.4f} roc_auc {roc_auc:.4f}pm{se_roc_auc:.4f} ap {ap_score:.4f}")

        # Compute ROC AUC and AP
        y_true = np.concatenate(y_true)
        num_pos, num_neg = np.sum(y_true), np.sum(y_true == 0)
        y_scores = np.concatenate(y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        se_roc_auc = se_auc(roc_auc, num_pos, num_neg)
        ap_score = average_precision_score(y_true, y_scores)

    logger.debug(f"Loss calculation took {t.interval:.3f}s")
    metrics = {
        f"eval/roc_auc_score/adv{adversarial_samples}": roc_auc,
        f"eval/se_roc_auc/adv{adversarial_samples}": se_roc_auc,
        f"eval/ap_score/adv{adversarial_samples}": ap_score,
        f"eval/num_examples/adv{adversarial_samples}": num_examples,
        f"eval/num_positive/adv{adversarial_samples}": np.sum(y_true),
    }

    if save_path:
        logger.info("Saving labels, scores and metrics to {}", save_path)
        torch.save({"y_true": y_true, "y_scores": y_scores, "metrics": metrics}, save_path)

    return (-roc_auc, metrics)


def train(
    run_name: str,
    # Data
    train_filepath: str = "data/codeclone/train_data.json",
    eval_filepath: str = "data/codeclone/valid_data.json",
    spm_filepath: str = "data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model",
    num_workers=1,
    max_seq_len=1024,
    max_eval_seq_len=1024,
    run_dir=RUN_DIR,
    balance_negatives=False,
    # Model
    resume_path: str = "",
    pretrain_resume_path: str = "",
    pretrain_resume_encoder_name: str = "encoder_q",  # encoder_q, encoder_k, encoder
    encoder_type: str = "transformer",
    n_encoder_layers: int = 6,
    d_model: int = 512,
    critic_type: str = "bilinear_identity",
    critic_bilinear_rank: int = None,
    # Optimization
    train_decoder_only: bool = False,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 8e-4,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.98,
    adam_eps: float = 1e-6,
    weight_decay: float = 0,
    warmup_steps: int = 5000,
    num_steps: int = 200000,
    # Evaluation
    save_every_steps: int = 5000,
    score_every_steps: int = 10,  # Interval for train ROC AUC, AP score
    evaluate_every_steps: int = 1250,
    # Augmentations
    subword_regularization_alpha: float = 0,
    # Computational
    use_cuda: bool = True,
    seed: int = 1,
):
    """Train model"""
    assert save_every_steps % evaluate_every_steps == 0

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
    train_negatives = CloneNegativesDataset(train_programs)
    train_dataset = torch.utils.data.ConcatDataset([train_positives, train_negatives])
    if balance_negatives:
        positive_weight = 1 / len(train_programs)
        negative_weight = 1 / len(train_negatives)
        weights = torch.cat([torch.zeros(len(train_programs)) + positive_weight, torch.zeros(len(train_negatives)) + negative_weight])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    else:
        sampler = None
    logger.info(f"Training dataset size: {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not balance_negatives,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=pad_collate,
        sampler=sampler,
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
        critic_type=critic_type,
        bilinear_rank=critic_bilinear_rank,
    )
    logger.info(f"Created CloneModel {encoder_type}, {critic_type} with {count_parameters(model)} params")

    # Load pretrained checkpoint
    if pretrain_resume_path:
        assert not resume_path
        logger.info(
            f"Resuming training from pretraining checkpoint {pretrain_resume_path}, pretrain_resume_encoder_name={pretrain_resume_encoder_name}"
        )
        checkpoint = torch.load(pretrain_resume_path)
        pretrained_state_dict = checkpoint["model_state_dict"]
        encoder_state_dict = {}
        # output_state_dict = {}
        assert pretrain_resume_encoder_name in ["encoder_k", "encoder_q", "encoder"]

        for key, value in pretrained_state_dict.items():
            if key.startswith(pretrain_resume_encoder_name + ".") and "project_layer" not in key:
                remapped_key = key[len(pretrain_resume_encoder_name + ".") :]
                logger.debug(f"Remapping checkpoint key {key} to {remapped_key}. Value mean: {value.mean().item()}")
                encoder_state_dict[remapped_key] = value
        model.encoder.load_state_dict(encoder_state_dict)
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

    # Initial evaluation
    logger.info(f"Evaluating model initially ({global_step} steps)...")
    eval_metric, eval_metrics = _evaluate(model, eval_loader, sp, pad_id=pad_id, use_cuda=use_cuda)
    for metric, value in eval_metrics.items():
        logger.info(f"Evaluation {metric} initial ({global_step} steps): {value:.4f}")
        max_eval_metrics[metric] = value
    eval_metrics["epoch"] = epoch
    wandb.log(eval_metrics, step=global_step)
    wandb.log({k + "_max": v for k, v in max_eval_metrics.items()}, step=global_step)

    for epoch in tqdm.trange(epoch + 1, num_epochs + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        model.train()
        model.module.encoder.eval()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for X, lengths, labels in pbar:
            if use_cuda:
                X, lengths, labels = X.cuda(), lengths.cuda(), labels.cuda()
            optimizer.zero_grad()
            similarity = model(X, lengths)  # B
            loss = F.binary_cross_entropy_with_logits(similarity, labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                train_metrics = {
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "train/labels_mean": labels.float().mean().item(),
                }

                # Compute scores in training batch
                if global_step % score_every_steps == 0:
                    y_true = labels.cpu().numpy()
                    y_scores = similarity.cpu().numpy()
                    roc_auc = roc_auc_score(y_true, y_scores)
                    ap_score = average_precision_score(y_true, y_scores)
                    train_metrics["train/roc_auc_score"] = roc_auc
                    train_metrics["train/ap_score"] = ap_score
                    pbar.set_description(f"epoch {epoch} loss {loss.item():.4f} roc_auc {roc_auc:.4f} ap {ap_score:.4f}")

                # Log loss
                global_step += 1
                wandb.log(train_metrics, step=global_step)

                # Evaluate
                if evaluate_every_steps and global_step % evaluate_every_steps == 0:
                    logger.info(f"Evaluating model after epoch {epoch} ({global_step} steps)...")
                    eval_metric, eval_metrics = _evaluate(model, eval_loader, sp, pad_id=pad_id, use_cuda=use_cuda)
                    for metric, value in eval_metrics.items():
                        logger.info(f"Evaluation {metric} after epoch {epoch} ({global_step} steps): {value:.4f}")
                        max_eval_metrics[metric] = max(value, max_eval_metrics[metric])
                    eval_metrics["epoch"] = epoch
                    wandb.log(eval_metrics, step=global_step)
                    wandb.log({k + "_max": v for k, v in max_eval_metrics.items()}, step=global_step)

                    # Save checkpoint
                    if save_every_steps and global_step % save_every_steps == 0 or eval_metric < min_eval_metric:
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
                            model_file = run_dir / f"ckpt_ep{epoch:04d}_step{global_step}.pth"
                        logger.info(f"Saving checkpoint to {model_file}...")
                        torch.save(checkpoint, str(model_file.resolve()))
                        logger.info("Done.")


def eval(
    # Data
    eval_filepath: str = "data/codeclone/test_data.json",
    spm_filepath: str = "data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model",
    save_path: str = None,  # path to save labels, similarity scores and metrics
    num_workers=1,  # num_loaders per loader.
    max_seq_len=-1,
    subsample_negatives=False,
    # Model
    resume_path: str = "",
    pretrain_resume_path: str = "",
    pretrain_resume_encoder_name: str = "encoder_q",  # encoder_q, encoder_k, encoder
    encoder_type: str = "transformer",
    n_encoder_layers: int = 6,
    d_model: int = 512,
    critic_type: str = "bilinear_identity",
    critic_bilinear_rank: int = None,
    edit_distance_mode=None,  # None, "strings", "tokens"
    # Optimization
    batch_size=16,
    adversarial_samples=0,  # 0 = don't use data augmentation.
                            # k > 0 = sample k augs for each program in a pair,
                            #   report minimum cosine similarity for pos pairs; max for neg pairs.
                            #   shuffle_loader must be False
    terser_module=False,  # Use when minifying an ES6 module. "use strict" is implied and names can be mangled on the top scope. If compress or mangle is enabled then the toplevel option will be enabled.
    terser_toplevel=False,  # set to true if you wish to enable top level variable and function name mangling and to drop unused variables and functions.
    # Loss
    subword_regularization_alpha: float = 0,
    # Computational
    use_cuda: bool = True,
    seed: int = 0,
    shuffle_loader: bool = False,
):
    """Evaluate model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = locals()
    logger.info(f"Config: {config}")

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")
    pad_collate = get_pad_collate(pad_id)

    # Create eval dataset and dataloader
    logger.info(f"Eval data path {eval_filepath}")
    augmentations = [
        {
            "fn": "terser",
            "options": {
                   "prob": 0.75,
                   "module": terser_module,
                   "toplevel": terser_toplevel
            }
        }
    ] if adversarial_samples > 0 else None
    eval_programs = CloneProgramsDataset(eval_filepath, sp, subword_regularization_alpha,
        augmentations=augmentations)
    eval_positives = ClonePositivesDataset(eval_programs)
    if subsample_negatives:
        eval_negatives = CloneNegativesDataset(eval_programs, max_pairs=len(eval_positives), seed=seed)
    else:
        eval_negatives = CloneNegativesDataset(eval_programs)
    eval_dataset = torch.utils.data.ConcatDataset([eval_positives, eval_negatives])
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    assert adversarial_samples == 0 or shuffle_loader is False  # don't shuffle because we make copies of the loader
                                                                # that need to access indices in the same order.
    eval_loaders = []
    for _ in range(max(1, adversarial_samples)):
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=shuffle_loader,
            num_workers=num_workers,
            collate_fn=pad_collate
        )
        eval_loaders.append(eval_loader)

    epoch = 0
    global_step = 0

    if edit_distance_mode:
        model = None
    else:
        # Create model
        model = CloneModel(
            n_tokens=sp.GetPieceSize(),
            pad_id=pad_id,
            encoder_type=encoder_type,
            n_encoder_layers=n_encoder_layers,
            d_model=d_model,
            critic_type=critic_type,
            bilinear_rank=critic_bilinear_rank,
        )
        logger.info(f"Created CloneModel {encoder_type}, {critic_type}")

        # Load pretrained checkpoint
        if pretrain_resume_path:
            assert not resume_path
            logger.info(
                f"Loading pretraining checkpoint {pretrain_resume_path}, pretrain_resume_encoder_name={pretrain_resume_encoder_name}"
            )
            checkpoint = torch.load(pretrain_resume_path)
            pretrained_state_dict = checkpoint["model_state_dict"]
            encoder_state_dict = {}
            assert pretrain_resume_encoder_name in ["encoder_k", "encoder_q", "encoder"]

            for key, value in pretrained_state_dict.items():
                if key.startswith(pretrain_resume_encoder_name + ".") and "project_layer" not in key:
                    remapped_key = key[len(pretrain_resume_encoder_name + ".") :]
                    logger.debug(f"Remapping checkpoint key {key} to {remapped_key}. Value mean: {value.mean().item()}")
                    encoder_state_dict[remapped_key] = value
            model.encoder.load_state_dict(encoder_state_dict)
            logger.info(f"Loaded state dict from {pretrain_resume_path}")

        # model = nn.DataParallel(model)
        model = model.cuda() if use_cuda else model

        if resume_path:
            assert not pretrain_resume_path
            logger.info(f"Resuming training from checkpoint {resume_path}")
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["epoch"]
            global_step = checkpoint["global_step"]

    # Evaluation
    logger.info(f"Evaluating model ({global_step} steps)...")
    if edit_distance_mode:
        _, eval_metrics = _evaluate_edit_distance(eval_loaders, sp, pad_id=pad_id,
            edit_distance_mode=edit_distance_mode, save_path=save_path)
    else:
        model.eval()
        _, eval_metrics = _evaluate(model, eval_loaders, sp, pad_id=pad_id,
            use_cuda=use_cuda, save_path=save_path)

    for metric, value in eval_metrics.items():
        logger.info(f"Evaluation {metric} initial ({epoch} epochs, {global_step} steps): {value:.4f}")


def split(data_path="data/codeclone/full_data.json", output_dir="data/codeclone/", valid_fraction=0.05, test_fraction=0.1, seed=1):
    """Split code clone data into train, validation and test sets by problem"""
    # Read problems and solutions, creating a list of lists
    all_problems = []
    with open(data_path, "r") as f:
        data = json.load(f)
        for _difficulty, problems in tqdm.tqdm(data.items(), desc="Reading problems and solutions"):
            for _problem_name, meta in problems.items():
                # Filter out None programs
                solutions = filter(lambda p: p, meta["srcs"])
                # Remove any exact duplicates
                solutions = list(set(solutions))
                if solutions:
                    all_problems.append(solutions)
    num_problems = len(all_problems)
    logger.info("Read {} problems, {} total solutions", num_problems, sum(map(len, all_problems)))

    # Split data
    np.random.seed(seed)
    indices = np.random.permutation(num_problems)
    num_test = int(num_problems * test_fraction)
    num_valid = int(num_problems * valid_fraction)
    num_train = num_problems - num_test - num_valid
    logger.info("Split: {} ({}%) train problems", num_train, num_train / num_problems)
    logger.info("Split: {} ({}%) valid problems", num_valid, num_valid / num_problems)
    test = [all_problems[i] for i in indices[:num_test]]
    valid = [all_problems[i] for i in indices[num_test : num_test + num_valid]]
    train = [all_problems[i] for i in indices[num_test + num_valid :]]
    logger.info("Split: {} ({}%) test problems, {} solutions", num_test, num_test / num_problems, sum(map(len, test)))
    assert len(test) == num_test
    assert len(valid) == num_valid
    assert len(train) == num_train

    # Write data
    def write(split_problems, split_name):
        output_path = os.path.join(output_dir, f"{split_name}_data.json")
        with open(output_path, "w") as out_f:
            json.dump(split_problems, out_f)
        logger.info("Wrote programs to {}", output_path)

    write(test, "test")
    write(valid, "valid")
    write(train, "train")


if __name__ == "__main__":
    fire.Fire({"train": train, "eval": eval, "split": split})
