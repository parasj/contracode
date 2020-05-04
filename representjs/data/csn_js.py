import gzip
import json
import jsonlines
import pathlib
import os
import re
from typing import List, Optional, Set, Iterable

from loguru import logger
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm

from representjs.data.util import dispatch_to_node

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


_fix_function_crop_regexes = [re.compile(r + r'(\s+|\()') for r in [
    r'\A^unction', r'\A^nction', r'\A^ction', r'\A^tion', r'\A^ion', r'\A^on', r'\A^n'
]]
_valid_identifier_regex = re.compile(r'^[a-zA-Z_$][0-9a-zA-Z_$]*$')
_num_invalid_id = 0
_num_valid_id = 0
def _make_example(json_dict, fields, require_fields):
    if require_fields:
        for field in require_fields:
            if field not in json_dict or not json_dict[field]:
                return None

    # Fix cropped "function" token at the begging of the function string
    for regex in _fix_function_crop_regexes:
        json_dict['function'] = regex.sub(r'function\1', json_dict['function'])

    if 'identifier' in json_dict and json_dict['identifier']:
        if require_fields is not None and 'identifier' in require_fields:
            # We need the identifier (method name) as a label. Filter invalid identifiers
            global _num_invalid_id, _num_valid_id
            if _valid_identifier_regex.match(json_dict['identifier']):
                _num_valid_id += 1
            else:
                # Skip this data point, it's not valid
                _num_invalid_id += 1
                return None

        # Remove function name from declaration, but leave it in the function body
        _function_name_regex = r'(function\s*)' + re.escape(json_dict['identifier'])
        replaced_fn = re.sub(_function_name_regex, r'\1', json_dict['function'])
        json_dict['function'] = replaced_fn

    return {out_key: json_dict[json_key] for json_key, out_key in fields.items()}


def _augment(transform_payload: List[dict]) -> List[str]:
    # Transform code
    transform_payload = json.dumps(transform_payload)
    stdout, stderr = dispatch_to_node('transform.js', transform_payload)
    if stderr:
        logger.error("WARNING: node error")
        logger.error("stdin: \n" + transform_payload)
        logger.error("stdout: \n" + stdout)
        logger.error("stderr: \n" + stderr)
    transformed = json.loads(stdout)
    assert isinstance(transformed, list)
    return transformed


class JSONLinesDataset(torch.utils.data.Dataset):
    """Defines a Dataset of columns stored in jsonlines format."""

    def __init__(self,
                 path,
                 fields=FUNCTION_ONLY_FIELDS,
                 require_fields: Optional[Iterable[str]] = None,
                 limit_size=-1, debug_charset=False,
                 **kwargs):
        """Create a JSONLinesDataset given a path and field mapping dictionary.
        Arguments:
            path (str): Path to the data file. Must be in .jsonl.gz or .jsonl format.
            fields (dict[str: str]):
                The keys should be a subset of the JSON keys,
                and the values should be desired names.
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON key names
                and also enables selecting a subset of columns to load.
            require_fields:
                Set of remapped data fields required to be present
        """
        label_char_set = set()
        nl = 0
        full_path = pathlib.Path(path).resolve()
        f = gzip.open(full_path, "rb") if path.endswith(".jsonl.gz") else full_path.open("r")
        reader = jsonlines.Reader(f)
        
        self.examples = []
        logger.debug(f"Loading {full_path}")
        for line in tqdm.tqdm(reader, desc=full_path.name, total=limit_size if limit_size >= 0 else None):
            example = _make_example(line, fields, require_fields)
            if example:
                self.examples.append(example)
                if 'label' in example.keys():
                    label_char_set.update(example['label'])
                if limit_size >= 0 and len(self.examples) >= limit_size:
                    print()
                    logger.info(f"WARNING: Limiting dataset size to {limit_size}")
                    break
            if debug_charset and len(label_char_set) != nl:
                logger.debug(f"update label char set: {label_char_set}")
                nl = len(label_char_set)
        f.close()

        logger.debug(f"Loaded {len(self.examples)} examples")
        if require_fields is not None and 'identifier' in require_fields:
            logger.debug("Num examples with valid identifier field:" + _num_valid_id)
            logger.debug("Num examples with invalid identifier field:" + _num_invalid_id)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def javascript_collate(examples: List[dict],
                       augmentations: List[dict],
                       sp: spm.SentencePieceProcessor,
                       program_mode: str,
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
    X = [torch.tensor([bos_id] + ids[:(max_length - 2)] + [eos_id]) for ids in X]
    X = pad_sequence(X, batch_first=True, padding_value=pad_id)

    # Create padded tensor for labels (good for seq2seq tasks)
    if "label" in examples[0]:
        label = [sp.EncodeAsIds(ex["label"]) for ex in examples]
        label = [torch.tensor([bos_id] + ids + [eos_id]) for ids in label]
        label = pad_sequence(label, batch_first=True, padding_value=pad_id)
    else:
        label = None

    if program_mode == "contrastive":
        # Reshape X to [B, 2, T]
        T = X.size(-1)
        X = torch.reshape(X, (2, B, -1))
        X = torch.transpose(X, 0, 1)
        assert X.shape == (B, 2, T)
        assert label is None, "label should be None when using contrastive program dataloader"

    return (X, label)


def javascript_dataloader(*args,
                          augmentations: List[dict],
                          sp: spm.SentencePieceProcessor,
                          program_mode: str = "identity",
                          subword_regularization_alpha: float = 0,
                          max_length: int = 1024,
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
    collate_fn = lambda batch: javascript_collate(batch, augmentations, sp, program_mode, subword_regularization_alpha, max_length)
    return torch.utils.data.DataLoader(*args, collate_fn=collate_fn, **kwargs)


if __name__ == "__main__":
    logger.info("===" * 10)
    logger.info("Test dataset")
    logger.info("===" * 10)
    data_dir = "data/codesearchnet_javascript"
    train_filepath = os.path.join(data_dir, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl")
    train_dataset = JSONLinesDataset(train_filepath, limit_size=100)
    logger.info(f"Number of training functions: {len(train_dataset)}")
    logger.info(f"Example {train_dataset[0]}")

    sp = spm.SentencePieceProcessor()
    sp.Load("data/codesearchnet_javascript/csnjs_8k_9995p_unigram.model")
    logger.info("===" * 10)
    logger.info("Test identity dataloader")
    logger.info("===" * 10)
    train_loader = javascript_dataloader(
        train_dataset, batch_size=2, shuffle=False,
        augmentations=[], sp=sp, program_mode="identity",
        subword_regularization_alpha=0.1)
    for X, label in train_loader:
        logger.info(f"X shape: {X.shape}")
        logger.info(f"Label: {label}")
        for i in range(len(X)):
            logger.info(f"Decoded X[{i}]: {sp.DecodeIds([int(id) for id in X[i]])}")
        break

    # TODO: Pass probability of applying each transform
    # augmentations = [{"fn": "rename_variable", "prob": 0.1}]
    # augmentations = [{"fn": "insert_var_declaration", "prob": 0.1}]
    augmentations = [{"fn": "sample_lines", "line_length_pct": 0.5}]
    logger.info("===" * 10)
    logger.info("Test augmentation dataloader")
    logger.info("===" * 10)
    train_loader = javascript_dataloader(
        train_dataset, batch_size=2, shuffle=False,
        augmentations=augmentations, sp=sp, program_mode="augmentation",
        subword_regularization_alpha=0.1)
    for X, label in train_loader:
        logger.info(f"X shape: {X.shape}")
        logger.info(f"Label: {label}")
        for i in range(len(X)):
            logger.info(f"Decoded X[{i}]: {sp.DecodeIds([int(id) for id in X[i]])}")
        break

    logger.info("===" * 10)
    logger.info("Test contrastive dataloader")
    logger.info("===" * 10)
    train_loader = javascript_dataloader(
        train_dataset, batch_size=2, shuffle=False,
        augmentations=augmentations, sp=sp, program_mode="contrastive",
        subword_regularization_alpha=0.1)
    for X, label in train_loader:
        logger.info(f"X shape: {X.shape}")
        logger.info(f"Label: {label}")
        for i in [0]:
            logger.info(f"##Transform 1: Decoded X[{i}, 0]:\n\t {sp.DecodeIds([int(id) for id in X[i, 0]])}")
            logger.info(f"##Transform 2: Decoded X[{i}, 1]:\n\t {sp.DecodeIds([int(id) for id in X[i, 1]])}")
        break

    ######### Test labeled tasks
    sp = spm.SentencePieceProcessor()
    sp.Load("data/codesearchnet_javascript/csnjs_8k_9995p_unigram.model")
    logger.info("===" * 10)
    logger.info("Test identity dataloader, method name labels")
    logger.info("===" * 10)
    labeled_dataset = JSONLinesDataset(train_filepath,
                                    fields={"function": "function", "identifier": "label"},
                                    require_fields=["identifier"], limit_size=32000, subword_regularization_alpha=0.1)
    logger.info(f"Len of labeled data {len(labeled_dataset)}")
    train_loader = javascript_dataloader(
        labeled_dataset, batch_size=2, shuffle=False,
        augmentations=[], sp=sp, program_mode="identity",
        subword_regularization_alpha=0.1)
    for X, label in train_loader:
        logger.info(f"X shape: {X.shape}")
        logger.info(f"Label: {label}")
        for i in range(len(X)):
            logger.info(f"Decoded X[{i}]: {sp.DecodeIds([int(id) for id in X[i]])}")
            logger.info(f"Decoded label[{i}]: {sp.DecodeIds([int(id) for id in label[i]])}")
            logger.info(f"Pieces for label[{i}]: {[sp.IdToPiece(int(id)) for id in label[i]]}")
        break
