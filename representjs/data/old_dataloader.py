import json
import os
from typing import List
import requests

import sentencepiece as spm
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from data.jsonl_dataset import JSONLinesDataset
from data.util import normalize_program
from data.util import dispatch_to_node


def _augment(transform_payload: List[dict]) -> List[str]:
    # Transform code
    transform_payload = json.dumps(transform_payload)
    is_successful, stdout, stderr = dispatch_to_node("transform.js", transform_payload)
    if is_successful:
        try:
            transformed = json.loads(stdout)
        except json.JSONDecodeError:
            # Transformation failed in Node.js (got malformed stdout), so don't transform
            logger.error(f"JSONDecodeError in _augment transform input: {transform_payload}")
            logger.error(f"JSONDecodeError in _augment transform stdout: {stdout}")
            logger.error(f"JSONDecodeError in _augment transform stderr: {stderr}")
            transformed = [prog["src"] for prog in transform_payload]
        except Exception as e:
            # Transformation failed in Node.js (got malformed stdout), so don't transform
            logger.error(f"Exception (maybe JSONDecodeError) in _augment: {e}")
            logger.error(f"Exception in _augment transform input: {transform_payload}")
            logger.error(f"Exception in _augment transform stdout: {stdout}")
            logger.error(f"Exception in _augment transform stderr: {stderr}")
            transformed = [prog["src"] for prog in transform_payload]
    else:
        logger.error("Non-zero exit code in _augment:")
        logger.error(f"Exception in _augment transform input: {transform_payload}")
        logger.error(f"Exception in _augment transform stdout: {stdout}")
        logger.error(f"Exception in _augment transform stderr: {stderr}")
        transformed = [prog["src"] for prog in transform_payload]
    assert isinstance(transformed, list)
    return transformed


_headers = {"Content-type": "application/json", "Accept": "application_json"}


def _augment_server(transform_payload: List[dict]) -> List[str]:
    # Transform code
    transform_payload = transform_payload
    response = None
    try:
        response = requests.post("http://127.0.0.1:3000", data=json.dumps(transform_payload), headers=_headers, timeout=5)
        assert response.status_code == 200
        transformed = response.json()
        assert isinstance(transformed, list)
    except Exception as e:
        # Transformation failed in Node.js (got malformed stdout), so don't transform
        logger.error(f"Exception in _augment_server: {e}")
        # logger.error(f"Exception in _augment_server transform input: {transform_payload}")
        # if response:
        #     logger.error(f"Exception in _augment_server transform status code: {response.status_code}")
        #     logger.error(f"Exception in _augment_server transform response: {response}")
        transformed = [prog["src"] for prog in transform_payload]
    return transformed


def get_javascript_collate(
    augmentations: List[dict],
    sp: spm.SentencePieceProcessor,
    program_mode: str,
    subword_regularization_alpha: float,
    max_length: int,
    max_target_length: int = 256,
):
    assert program_mode in ["contrastive", "augmentation", "identity"]
    bos_id = sp.PieceToId("<s>")
    eos_id = sp.PieceToId("</s>")
    pad_id = sp.PieceToId("[PAD]")

    def javascript_collate(examples: List[dict]):
        """Augments and batches a list of function dicts.

        Arguments:
            examples (List[dict[str, Any]]). The dicts must have key "function".
            augmentations (List[dict]). Augmentations to apply to the functions.
                example: [{"fn": "extract_methods"}]
            sp (SentencePieceProcessor): For tokenizing batch elements after augmentations
        """
        B = len(examples)
        if program_mode in ["contrastive", "augmentation"]:
            # Set up transformation input
            transform_payload = []
            for example in examples:
                transform_payload.append(dict(src=example["function"], augmentations=augmentations))
            if program_mode == "contrastive":
                # Augment each input function twice
                transform_payload = transform_payload + transform_payload
            X = _augment_server(transform_payload)
        else:
            X = [prog["function"] for prog in examples]

        # Normalize programs
        X = [normalize_program(prog) for prog in X]

        # Encode as ids with sentencepiece
        if subword_regularization_alpha:
            # using subword regularization: https://arxiv.org/pdf/1804.10959.pdf
            # NOTE: what is the second argument here (-1)?
            X = [sp.SampleEncodeAsIds(prog, -1, subword_regularization_alpha) for prog in X]
        else:
            # using the best decoding
            X = [sp.EncodeAsIds(prog) for prog in X]

        # Create padded tensor for batch, [B, T] or [2B, T]
        X = [torch.tensor([bos_id] + ids[: (max_length - 2)] + [eos_id]) for ids in X]
        X_lengths = torch.tensor([len(x) for x in X], dtype=torch.long)
        X = pad_sequence(X, batch_first=True, padding_value=pad_id)

        # Create padded tensor for labels (good for seq2seq tasks)
        if "label" in examples[0]:
            label = [sp.EncodeAsIds(ex["label"]) for ex in examples]
            label = [torch.tensor([bos_id] + ids[: (max_target_length - 2)] + [eos_id]) for ids in label]
            label_lengths = torch.tensor([len(label_item) for label_item in label], dtype=torch.long)
            label = pad_sequence(label, batch_first=True, padding_value=pad_id)
        else:
            label = None
            label_lengths = None

        if program_mode == "contrastive":
            # Reshape X to [B, 2, T]
            T = X.size(-1)
            X = torch.reshape(X, (2, B, -1))
            X = torch.transpose(X, 0, 1)
            assert X.shape == (B, 2, T)
            X_lengths = torch.reshape(X_lengths, (2, B))
            assert label is None, "label should be None when using contrastive program dataloader"
        return (X, label, X_lengths, label_lengths)

    return javascript_collate


def javascript_dataloader(
    *args,
    augmentations: List[dict],
    sp: spm.SentencePieceProcessor,
    program_mode: str = "identity",
    subword_regularization_alpha: float = 0,
    max_length: int = 1024,
    max_target_length: int = 256,
    spm_unigram_path: str = None,
    **kwargs,
):
    """
    Arguments:
        program_mode
            program_mode="contrastive": Batches are (LongTensor[B, 2, max_seq_len], label)
            program_mode="augmentation" or "none": Batches are (LongTensor[B, max_seq_len], label)
        sp: Vocabulary. Example of creating sp:
            sp = spm.SentencePieceProcessor()
            sp.Load("data/codesearchnet_javascript/csnjs_8k_9995p_unigram.model")
    """
    assert "collate_fn" not in kwargs
    if sp is None:
        sp = spm.SentencePieceProcessor()
        sp.load(spm_unigram_path)
    collate_fn = get_javascript_collate(
        augmentations, sp, program_mode, subword_regularization_alpha, max_length, max_target_length=max_target_length
    )
    return torch.utils.data.DataLoader(*args, collate_fn=collate_fn, **kwargs)


if __name__ == "__main__":
    logger.debug("Loading unigram encoding")
    sp = spm.SentencePieceProcessor()
    sp.Load("data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model")

    augmentation_examples = [
        {"fn": "insert_var_declaration", "prob": 0.1},
        {"fn": "rename_variable", "prob": 0.1},
        # 1 - .9^3 chance of at least one of compress, mangle, and compress_mangle being applied
        # {"fn": "compress", "prob": 0.1},
        # {"fn": "mangle", "prob": 0.1},
        # {"fn": "compress_mangle", "prob": 0.1},
        # {"fn": "remove_comments", "prob": 0.2},
        {"fn": "terser", "prob": 0.5, "prob_mangle": 0.1},
        {"fn": "sample_lines", "line_length_pct": 0.9},
    ]
    for augmentation in augmentation_examples:
        logger.info(f"Testing augmentation {augmentation}")
        data_dir = "data/codesearchnet_javascript"
        train_filepath = os.path.join(data_dir, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl")
        train_dataset = JSONLinesDataset(train_filepath, limit_size=5)
        train_loader = javascript_dataloader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            augmentations=[augmentation],
            sp=sp,
            program_mode="augmentation",
            subword_regularization_alpha=0.1,
        )
        for X, label in train_loader:
            logger.info(f"X shape: {X.shape}")
            logger.info(f"Label: {label}")

    # logger.info("===" * 10)
    # logger.info("Test dataset")
    # logger.info("===" * 10)
    # data_dir = "data/codesearchnet_javascript"
    # train_filepath = os.path.join(data_dir, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl")
    # train_dataset = JSONLinesDataset(train_filepath, limit_size=100)
    # logger.info(f"Number of training functions: {len(train_dataset)}")
    # logger.info(f"Example {train_dataset[0]}")
    #
    # sp = spm.SentencePieceProcessor()
    # sp.Load("data/codesearchnet_javascript/csnjs_8k_9995p_unigram.model")
    # logger.info("===" * 10)
    # logger.info("Test identity dataloader")
    # logger.info("===" * 10)
    # train_loader = javascript_dataloader(
    #     train_dataset, batch_size=2, shuffle=False,
    #     augmentations=[], sp=sp, program_mode="identity",
    #     subword_regularization_alpha=0.1)
    # for X, label in train_loader:
    #     logger.info(f"X shape: {X.shape}")
    #     logger.info(f"Label: {label}")
    #     for i in range(len(X)):
    #         logger.info(f"Decoded X[{i}]: {sp.DecodeIds([int(id) for id in X[i]])}")
    #     break
    #
    # # TODO: Pass probability of applying each transform
    # # augmentations = [{"fn": "rename_variable", "prob": 0.1}]
    # # augmentations = [{"fn": "insert_var_declaration", "prob": 0.1}]
    # augmentations = [{"fn": "sample_lines", "line_length_pct": 0.5}]
    # logger.info("===" * 10)
    # logger.info("Test augmentation dataloader")
    # logger.info("===" * 10)
    # train_loader = javascript_dataloader(
    #     train_dataset, batch_size=2, shuffle=False,
    #     augmentations=augmentations, sp=sp, program_mode="augmentation",
    #     subword_regularization_alpha=0.1)
    # for X, label in train_loader:
    #     logger.info(f"X shape: {X.shape}")
    #     logger.info(f"Label: {label}")
    #     for i in range(len(X)):
    #         logger.info(f"Decoded X[{i}]: {sp.DecodeIds([int(id) for id in X[i]])}")
    #     break
    #
    # logger.info("===" * 10)
    # logger.info("Test contrastive dataloader")
    # logger.info("===" * 10)
    # train_loader = javascript_dataloader(
    #     train_dataset, batch_size=2, shuffle=False,
    #     augmentations=augmentations, sp=sp, program_mode="contrastive",
    #     subword_regularization_alpha=0.1)
    # for X, label in train_loader:
    #     logger.info(f"X shape: {X.shape}")
    #     logger.info(f"Label: {label}")
    #     for i in [0]:
    #         logger.info(f"##Transform 1: Decoded X[{i}, 0]:\n\t {sp.DecodeIds([int(id) for id in X[i, 0]])}")
    #         logger.info(f"##Transform 2: Decoded X[{i}, 1]:\n\t {sp.DecodeIds([int(id) for id in X[i, 1]])}")
    #     break
    #
    # ######### Test labeled tasks
    # sp = spm.SentencePieceProcessor()
    # sp.Load("data/codesearchnet_javascript/csnjs_8k_9995p_unigram.model")
    # logger.info("===" * 10)
    # logger.info("Test identity dataloader, method name labels")
    # logger.info("===" * 10)
    # labeled_dataset = JSONLinesDataset(train_filepath,
    #                                 fields={"function": "function", "identifier": "label"},
    #                                 require_fields=["identifier"], limit_size=32000, subword_regularization_alpha=0.1)
    # logger.info(f"Len of labeled data {len(labeled_dataset)}")
    # train_loader = javascript_dataloader(
    #     labeled_dataset, batch_size=2, shuffle=False,
    #     augmentations=[], sp=sp, program_mode="identity",
    #     subword_regularization_alpha=0.1)
    # for X, label in train_loader:
    #     logger.info(f"X shape: {X.shape}")
    #     logger.info(f"Label: {label}")
    #     for i in range(len(X)):
    #         logger.info(f"Decoded X[{i}]: {sp.DecodeIds([int(id) for id in X[i]])}")
    #         logger.info(f"Decoded label[{i}]: {sp.DecodeIds([int(id) for id in label[i]])}")
    #         logger.info(f"Pieces for label[{i}]: {[sp.IdToPiece(int(id)) for id in label[i]]}")
    #     break
