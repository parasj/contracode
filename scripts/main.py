import os

import fire
from loguru import logger
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb

from representjs import RUN_DIR
from representjs.data.csn_js import javascript_dataloader, JSONLinesDataset
from representjs.models import TransformerModel

# Default argument values
DATA_DIR = "data/codesearchnet_javascript"
CSNJS_TRAIN_FILEPATH = os.path.join(DATA_DIR, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
SPM_UNIGRAM_FILEPATH = os.path.join(DATA_DIR, "csnjs_8k_9995p_unigram.model")


def train(
        run_name: str,

        # Data
        train_filepath: str = CSNJS_TRAIN_FILEPATH,
        spm_filepath: str = SPM_UNIGRAM_FILEPATH,
        program_mode="identity",
        label_mode="none",
        num_workers=1,
        limit_dataset_size=-1,

        # Optimization
        num_epochs: int = 100,
        save_every: int = 2,
        batch_size: int = 256,
        lr: float = 8e-4,

        # Loss
        subword_regularization_alpha: float = 0,

        # Computational
        use_cuda: bool = True
):
    """Train model"""
    run_dir = RUN_DIR / run_name
    run_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving model checkpoints to {run_dir}")
    config = locals()
    print(config)
    wandb.init(name=run_name, config=config, job_type="training", project="code-representation", entity="ml4code")

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"

    augmentations = [{"fn": "sample_lines", "line_length_pct": 0.5}]

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")

    # Create dataset and dataloader
    if label_mode == "identifier":
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
    print("Training dataset size:", len(train_dataset))
    train_loader = javascript_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        augmentations=augmentations, sp=sp, program_mode=program_mode,
        subword_regularization_alpha=subword_regularization_alpha)

    model = TransformerModel(ntoken=sp.GetPieceSize(), ninp=512, dropout=0.1)
    model = nn.DataParallel(model)
    model = model.cuda() if use_cuda else model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     lr,
    #     epochs=num_epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.02,  # Warm up for 2% of the total training time
    # )

    global_step = 0
    for epoch in tqdm.trange(1, num_epochs + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        wandb.log({
            'epoch': epoch,
            # 'lr': scheduler.get_lr()
        })
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for X, Y in pbar:
            # TODO: Implement pretraining
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda() if Y is not None else Y
            optimizer.zero_grad()
            logits = model(X, Y, pad_id=pad_id)
            loss = F.cross_entropy(logits.transpose(1, 2), Y, ignore_index=pad_id)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Log loss
            global_step += 1
            wandb.log({
                "epoch": epoch,
                f"label-{label_mode}/program-{program_mode}/loss": loss.item()
            }, step=global_step)
            pbar.set_description(f"epoch {epoch} loss {loss.item():.4f}")

        if epoch % save_every == 0:
            model_file = run_dir / f"ckpt_ep{epoch:04d}.pth"
            logger.info(f"Saving checkpoint to {model_file}...", endl=" ")
            torch.save({
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config
            }, str(model_file.resolve()))
            logger.info("Done.")


if __name__ == "__main__":
    fire.Fire({"train": train})
