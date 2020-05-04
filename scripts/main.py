import os
from typing import List

import sentencepiece as spm
import torch
import tqdm
import wandb

from representjs.data.csn_js import javascript_dataloader


# Default argument values
DATA_DIR = "data/codesearchnet_javascript"
TRAIN_FILEPATH = os.path.join(DATA_DIR, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl")
SPM_FILEPATH = os.path.join(DATA_DIR, "csnjs_8k_9995p_unigram.model")

# Constants
ALL_RUN_DIR = "data/runs"


def train(
    run_name: str,
    # Data
    train_filepath: str=TRAIN_FILEPATH,
    spm_filepath: str=SPM_FILEPATH,
    program_mode="identity",
    label_mode="none",
    # Optimization
    num_epochs: int=100,
    save_every: int=5,
    batch_size: int=256,
    lr: float=8e-4,
    # Loss
    subword_regularization_alpha: float=0,
    # Computational
    use_cuda: bool=True):
    """Train model"""
    run_dir = os.path.join(ALL_RUN_DIR, run_name)
    os.makedirs(run_dir)
    print(f"Saving model checkpoints to {run_dir}")
    config = locals()
    wandb.init(name=run_name, config=config, job_type="training", project="code-representation")

    augmentations = [{"fn": "sample_lines", "line_length_pct": 0.5}]

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)

    # Create dataset and dataloader
    if label_mode == "method_name":
        dataset_fields = {"function": "function", "identifier": "label"}
        dataset_require_fields = ["identifier"]
    elif label_mode == "docstring":
        dataset_fields = {"function": "function", "docstring": "label"}
        dataset_require_fields = ["docstring"]
    else:
        dataset_fields = {"function": "function"}
        dataset_require_fields = []
    train_dataset = JSONLinesDataset(train_filepath,
                                     fields=dataset_fields,
                                     require_fields=dataset_require_fields,
                                     limit_size=limit_dataset_size)

    train_loader = javascript_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True,
        augmentations=augmentations, sp=sp, program_mode=program_mode,
        label_mode=label_mode, subword_regularization_alpha=subword_regularization_alpha)

    model = torch.nn.Transformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.02,  # Warm up for 2% of the total training time
    )

    global_step = 0
    for epoch in tqdm.trange(1, num_epochs+1, desc="training", unit="epoch"):
        print(f"Starting epoch {epoch}")

        for X, _ in tqdm.tqdm(train_loader, desc=f"epoch {epoch}"):
            X = X.cuda() if use_cuda else X
            optimizer.zero_grad()
            # import IPython
            # IPython.embed()
            # import sys
            # sys.exit()
            loss = None# TODO
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            global_step += 1
            wandb.log({
                "epoch": epoch,
                f"pm-{program_mode}/lm-{label_mode}/loss": loss.item()
            }, step=global_step)
        
        if epoch % save_every == 0:
            model_file = os.path.join(run_dir, f"ckpt_ep{epoch:04d}.pth")
            print(f"Saving checkpoint to {model_file}...", endl=" ")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config
            }, model_file)
            print("Done.")

if __name__=="__main__":
    fire({
        "train": train
    })
