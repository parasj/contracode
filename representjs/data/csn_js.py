import gzip
import io
import json
import jsonlines
import os
import re
from typing import List

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm

from representjs.data.transforms.util import dispatch_to_node

# Possible keys:
#   'identifier' for method name which may be blank for anonymous fns
#   'function' for the function as a string
#   'function_tokens'
#   'docstring' for the docstring (blank for most, filled in for about 100k)
#   'docstring_tokens'
FUNCTION_ONLY_FIELDS = {"function": "function"}


_newline_regex = re.compile(r'\n')
_whitespace_regex = re.compile(r'[ \t\n]+')
def normalize_program(fn: str):
    fn = _newline_regex.sub(r' [EOL]', fn)
    fn = _whitespace_regex.sub(' ', fn)
    return fn


def _make_example(json_dict, fields):
    out_dict = {}
    for json_key, out_key in fields.items():
        out_dict[out_key] = json_dict[json_key]
    return out_dict


def _augment(transform_payload: List[dict]) -> List[str]:
    # Transform code
    transform_payload = json.dumps(transform_payload)
    stdout, stderr = dispatch_to_node('transform.js', transform_payload)
    if stderr:
        print("===" * 10)
        print("WARNING: node error:")
        print("------- stdin")
        print(transform_payload)
        print("------- stdout")
        print(stdout)
        print("------- stderr")
        print(stderr)
        print("===" * 10)
    transformed = json.loads(stdout)
    assert isinstance(transformed, list)
    return transformed


class JSONLinesDataset(torch.utils.data.Dataset):
    """Defines a Dataset of columns stored in jsonlines format."""

    def __init__(self, path, fields=FUNCTION_ONLY_FIELDS, **kwargs):
        """Create a JSONLinesDataset given a path and field mapping dictionary.
        Arguments:
            path (str): Path to the data file. Must be in .jsonl.gz or .jsonl format.
            fields (dict[str: str]):
                The keys should be a subset of the JSON keys,
                and the values should be desired names.
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON key names
                and also enables selecting a subset of columns to load.
        """
        if path.endswith(".jsonl.gz"):
            f = gzip.open(os.path.expanduser(path), "rb")
        else:
            f = open(os.path.expanduser(path), "r")
        reader = jsonlines.Reader(f)
        self.examples = []
        for line in tqdm.tqdm(reader, f"Loading {path}"):
            self.examples.append(_make_example(line, fields))
            if len(self.examples) >= 100:
                break
        f.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def javascript_collate(examples: List[dict],
                       augmentations: List[dict],
                       sp: spm.SentencePieceProcessor,
                       program_mode: str,
                       label_mode: str,
                       subword_regularization_alpha: float,
                       max_length: int):
    """Augments and batches a list of function dicts.

    Arguments:
        examples (List[dict[str, Any]]). The dicts must have key "function".
        augmentations (List[dict]). Augmentations to apply to the functions.
            example: [{"fn": "extract_methods"}]
        sp (SentencePieceProcessor): For tokenizing batch elements after augmentations
    """
    assert program_mode in ["contrastive", "augmentation", "identity"]
    assert label_mode == "none"

    B = len(examples)

    if program_mode in ["contrastive", "augmentation"]:
        # Set up transformation input
        transform_payload = []
        for example in examples:
            transform_payload.append(dict(
                src=example["function"],
                augmentations=augmentations))
        if program_mode == "contrastive":
            # Augment each input function twice
            transform_payload = transform_payload + transform_payload
        X = _augment(transform_payload)
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
    bos_id = sp.PieceToId("<s>")
    eos_id = sp.PieceToId("</s>")
    pad_id = sp.PieceToId("[PAD]")
    X = [torch.tensor([bos_id] + ids[:(max_length-2)] + [eos_id]) for ids in X]
    X = pad_sequence(X, batch_first=True, padding_value=pad_id)
    
    if program_mode == "contrastive":
        # Reshape X to [B, 2, T]
        T = X.size(-1)
        X = torch.reshape(X, (2, B, -1))
        X = torch.transpose(X, 0, 1)
        assert X.shape == (B, 2, T)

    # TODO: Add labels
    return (X, None)


def javascript_dataloader(*args,
                          augmentations: List[dict],
                          sp: spm.SentencePieceProcessor,
                          program_mode: str="identity",
                          label_mode: str="none",
                          subword_regularization_alpha: float=0,
                          max_length: int=1024,
                          **kwargs):
    """
    Arguments:
        program_mode
            program_mode="contrastive": Batches are (LongTensor[B, 2, max_seq_len], label)
            program_mode="augmentation" or "none": Batches are (LongTensor[B, max_seq_len], label)
        sp: Vocabulary. Example of creating sp:
            sp = spm.SentencePieceProcessor()
            sp.Load("data/codesearchnet_javascript/csnjs_8k_9995p_unigram.model")
    """
    assert 'collate_fn' not in kwargs
    collate_fn = lambda batch: javascript_collate(batch, augmentations, sp, program_mode, label_mode, subword_regularization_alpha, max_length)
    return torch.utils.data.DataLoader(*args, collate_fn=collate_fn, **kwargs)


if __name__ == "__main__":
    print("===" * 10)
    print("Test dataset")
    print("===" * 10)
    data_dir = "data/codesearchnet_javascript"
    train_filepath = os.path.join(data_dir, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl")
    train_dataset = JSONLinesDataset(train_filepath)
    print("Number of training functions:", len(train_dataset[0]))
    print("Example", train_dataset[0])
    print()

    sp = spm.SentencePieceProcessor()
    sp.Load("data/codesearchnet_javascript/csnjs_8k_9995p_unigram.model")
    print("===" * 10)
    print("Test identity dataloader")
    print("===" * 10)
    train_loader = javascript_dataloader(
        train_dataset, batch_size=2, shuffle=False,
        augmentations=[], sp=sp, program_mode="identity", label_mode="none",
        subword_regularization_alpha=0.1)
    for X, label in train_loader:
        print("X shape:", X.shape)
        print("Label:", label)
        for i in range(len(X)):
            print(f"Decoded X[{i}]:", sp.DecodeIds([int(id) for id in X[i]]))
            print()
        break

    # TODO: Pass probability of applying each transform
    # augmentations = [{"fn": "rename_variable", "prob": 0.1}]
    # augmentations = [{"fn": "insert_var_declaration", "prob": 0.1}]
    augmentations = [{"fn": "sample_lines", "line_length_pct": 0.5}]
    print("===" * 10)
    print("Test augmentation dataloader")
    print("===" * 10)
    train_loader = javascript_dataloader(
        train_dataset, batch_size=2, shuffle=False,
        augmentations=augmentations, sp=sp, program_mode="augmentation", label_mode="none",
        subword_regularization_alpha=0.1)
    for X, label in train_loader:
        print("X shape:", X.shape)
        print("Label:", label)
        for i in range(len(X)):
            print(f"Decoded X[{i}]:", sp.DecodeIds([int(id) for id in X[i]]))
            print()
        break

    print("===" * 10)
    print("Test contrastive dataloader")
    print("===" * 10)
    train_loader = javascript_dataloader(
        train_dataset, batch_size=2, shuffle=False,
        augmentations=augmentations, sp=sp, program_mode="contrastive", label_mode="none",
        subword_regularization_alpha=0.1)
    for X, label in train_loader:
        print("X shape:", X.shape)
        print("Label:", label)
        for i in [0]:
            print(f"##Transform 1: Decoded X[{i}, 0]:\n\t", sp.DecodeIds([int(id) for id in X[i, 0]]))
            print(f"##Transform 2: Decoded X[{i}, 1]:\n\t", sp.DecodeIds([int(id) for id in X[i, 1]]))
            print()
        break
