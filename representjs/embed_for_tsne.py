import gzip
import pathlib
import pickle
import random
import os

import fire
import jsonlines
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import tqdm
from loguru import logger

from models.code_moco import CodeMoCo
from models.encoder import CodeEncoder, CodeEncoderLSTM
from representjs import CSNJS_DIR, DATA_DIR
from representjs.type_prediction import count_parameters
from data.precomputed_dataset import PrecomputedDataset

DEFAULT_CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
DEFAULT_SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


def embed_coco(checkpoint, data_path, spm_filepath=DEFAULT_SPM_UNIGRAM_FILEPATH, encoder_type="transformer", n_encoder_layers=6):
    # read data
    df = pd.read_pickle(data_path)

    # read tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    n_tokens = sp.GetPieceSize()
    pad_id = sp.PieceToId("[PAD]")

    # load model
    model = CodeMoCo(
        n_tokens=n_tokens,
        pad_id=pad_id,
        encoder_config=dict(encoder_type=encoder_type, n_encoder_layers=n_encoder_layers, project="hidden"),
    )
    state = torch.load(checkpoint)
    print(state["model_state_dict"].keys())
    model.load_state_dict(state["model_state_dict"])
    model.cuda()
    model.eval()

    out_rows = []
    with torch.no_grad():
        for row_idx in tqdm.tqdm(list(range(len(df))), desc="Table"):
            text = df.loc[row_idx]["code"]
            func_name = df.loc[row_idx]["func_name"]
            x_encoded = torch.LongTensor(sp.EncodeAsIds(text)).cuda()
            lens = torch.LongTensor([len(x_encoded)])
            try:
                embed_x = model.embed_x(x_encoded.unsqueeze(0), lens).cpu().numpy()
                out_rows.append(dict(code=text, func_name=func_name, embedding=embed_x))
            except Exception as e:
                print("Error!", e)
                continue

    tsne_out_path = DATA_DIR / "tsne"
    tsne_out_path.mkdir(parents=True, exist_ok=True)
    print("writing output to ", tsne_out_path.resolve())
    with (tsne_out_path / "tsne_out_embedded_grouped_hidden.pickle").open("wb") as f:
        pickle.dump(out_rows, f)


def embed_augmented(
    # Data
    data_filepath: str,
    output_dir: str,
    spm_filepath: str,
    num_workers=1,
    max_seq_len=-1,
    min_alternatives=2,
    # Model
    encoder_type: str = "lstm",
    pretrain_resume_path: str = "",
    pretrain_resume_encoder_name: str = "encoder_q",  # encoder_q, encoder_k, encoder
    pretrain_resume_project: bool = False,
    # no_output_attention: bool = False,
    n_encoder_layers: int = 2,
    d_model: int = 512,
    # Loss
    subword_regularization_alpha: float = 0,
    # Computational
    use_cuda: bool = True,
    seed: int = 0,
):
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
    mask_id = sp.PieceToId("[MASK]")

    # Create model
    if encoder_type == "lstm":
        encoder = CodeEncoderLSTM(
            n_tokens=sp.GetPieceSize(),
            d_model=d_model,
            d_rep=256,
            n_encoder_layers=n_encoder_layers,
            dropout=0.1,
            pad_id=pad_id,
            project=False,
        )
        encoder.config["project"] = "hidden"
        logger.info(f"Created CodeEncoderLSTM with {count_parameters(encoder)} params")
    elif encoder_type == "transformer":
        encoder = CodeEncoder(sp.GetPieceSize(), d_model, 256, 8, n_encoder_layers, 2048, 0.1, "relu", True, pad_id, project=False)
        logger.info(f"Created CodeEncoder with {count_parameters(encoder)} params")

    # Load pretrained checkpoint
    if pretrain_resume_path:
        logger.info(
            f"Resuming training from pretraining checkpoint {pretrain_resume_path}, pretrain_resume_encoder_name={pretrain_resume_encoder_name}"
        )
        checkpoint = torch.load(pretrain_resume_path)
        pretrained_state_dict = checkpoint["model_state_dict"]

        for key in pretrained_state_dict.keys():
            print("Pretrained state dict:", key)
        for key in encoder.state_dict().keys():
            print("Encoder state dict:", key)

        encoder_state_dict = {}
        assert pretrain_resume_encoder_name in ["encoder_k", "encoder_q", "encoder"]

        for key, value in pretrained_state_dict.items():
            if key.startswith(pretrain_resume_encoder_name + ".") and "project_layer" not in key:
                remapped_key = key[len(pretrain_resume_encoder_name + ".") :]
                logger.debug(f"Remapping checkpoint key {key} to {remapped_key}. Value mean: {value.mean().item()}")
                encoder_state_dict[remapped_key] = value
        encoder.load_state_dict(encoder_state_dict)
        logger.info(f"Loaded state dict from {pretrain_resume_path}")

    # Parallelize across GPUs
    encoder = nn.DataParallel(encoder)
    encoder = encoder.cuda() if use_cuda else encoder

    # Load batches consisting of augmented variants of the same program
    sp = spm.SentencePieceProcessor()
    sp.Load(config["spm_filepath"])
    pad_id = sp.PieceToId("[PAD]")

    def pad_collate(batch):
        assert len(batch) == 1
        X = batch[0]
        B = len(X)

        # Create tensor of sequence lengths, [B] or [2B]
        lengths = torch.tensor([len(x) for x in X], dtype=torch.long)

        # Create padded tensor for batch, [B, T]
        X = pad_sequence(X, batch_first=True, padding_value=pad_id)

        return X, lengths

    dataset = PrecomputedDataset(
        data_filepath,
        min_alternatives=min_alternatives,
        program_mode="all_alternatives",
        limit_size=-1,
        sp=sp,
        subword_regularization_alpha=subword_regularization_alpha,
        max_length=max_seq_len,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=pad_collate, num_workers=num_workers, drop_last=False, pin_memory=False,
    )

    representations = []
    encoder.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        # Evaluate metrics
        logger.info(f"Evaluating encoder...")
        pbar = tqdm.tqdm(loader, desc="evalaute")
        for X, lengths in pbar:
            rep = encoder(X.cuda(), lengths.cuda(), None)  # [B, n_layers*n_directions*d_model]
            if encoder_type == "transformer":
                rep = rep[0]
                assert len(rep.shape) == 3
                # rep = rep.mean(dim=0)  # rep is [T, B, dimension], so take mean across sequence
                rep, _ = rep.max(dim=0)  # Max pool
            rep = rep.cpu().numpy()
            X = X.cpu().numpy()
            print("rep", type(rep), "X", type(X))
            print("rep", rep.shape, "X", X.shape)
            representations.append((X, rep))

            if len(representations) and len(representations) % 100 == 0:
                path = os.path.join(output_dir, f"tokens_and_embeddings_{len(representations):06d}.pth")
                logger.info(f"Saving representations to {path}")
                # with open(path, "wb") as f:
                #     pickle.dump(representations, f)
                # torch.save(path, representations)
                torch.save(representations, path)
                # np.save(path, representations)


def filter_augmented_dataset(data_path, out_path, min_alternatives):
    full_path = pathlib.Path(data_path).resolve()
    read_f = gzip.open(full_path, "rb") if data_path.endswith(".gz") else full_path.open("r")
    data = pickle.load(read_f)
    read_f.close()

    full_out_path = pathlib.Path(out_path).resolve()
    write_f = gzip.open(full_out_path, "wb") if out_path.endswith(".gz") else full_out_path.open("w")
    logger.debug(f"Writing output to {full_out_path}...")

    logger.debug(f"Loading {full_path}")
    total_lines = 0
    num_written = 0
    output = []
    for alternatives in tqdm.tqdm(data, desc=full_path.name):
        total_lines += 1
        if len(alternatives) >= min_alternatives:
            output.append(alternatives)
            num_written += 1

        if total_lines % 1000 == 0:
            logger.debug(f"Filtered jsonl to {num_written}/{total_lines}")
    logger.debug(f"DONE: Filtered jsonl to {num_written}/{total_lines}")
    pickle.dump(output, write_f, protocol=pickle.HIGHEST_PROTOCOL)
    write_f.close()


if __name__ == "__main__":
    fire.Fire({"embed_coco": embed_coco, "embed_augmented": embed_augmented, "filter_augmented_dataset": filter_augmented_dataset})
