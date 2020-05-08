import gzip
import json
import jsonlines
import io
import os
from typing import List
import re

import pathlib
import fire
import sentencepiece as spm
import torch
import tqdm
from loguru import logger
import sentencepiece as spm

from representjs.data.csn_js_loader import normalize_program, _augment
from representjs.data.csn_js_jsonl import JSONLinesDataset
from representjs.data.util import dispatch_to_node

_valid_identifier_regex = re.compile(r'^[a-zA-Z_$][0-9a-zA-Z_$]*$')

def filter_dataset(path: str, out_path: str, require_fields=[], exclude_transform_errors=False):
    logger.debug(f"Requiring fields {require_fields}")
    full_path = pathlib.Path(path).resolve()
    f = gzip.open(full_path, "rb") if path.endswith(".jsonl.gz") else full_path.open("r")
    reader = jsonlines.Reader(f)

    total_lines = 0
    examples = []
    logger.debug(f"Loading {full_path}")
    for json_dict in tqdm.tqdm(reader, desc=full_path.name):
        total_lines += 1
        # Check all required fields are present
        if any([field not in json_dict or not json_dict[field] for field in require_fields]):
            continue

        # We need the identifier (method name) as a label. Filter invalid identifiers
        if 'identifier' in require_fields and _valid_identifier_regex.match(json_dict['identifier']) == None:
            continue

        # Try to parse/transform the code, and filter out if we can't
        if exclude_transform_errors:
            # Set up transformation input
            transform_payload = [dict(
                src=json_dict["function"],  # TODO: this key should be "code" for supervised set
                augmentations=[{"fn": "rename_variable", "prob": 1.0}]
            )]
            transform_payload = json.dumps(transform_payload)
            stdout, stderr = dispatch_to_node('transform.js', transform_payload)
            if stderr:
                continue
            # X = _augment(transform_payload)

        examples.append(json_dict)
        if total_lines % 1 == 0:
            logger.debug(f"Filtered jsonl to {len(examples)}/{total_lines}")
    f.close()

    logger.debug(f"DONE: Filtered jsonl to {len(examples)}/{total_lines}")

    # TODO: Subsample
    
    # Write output
    full_out_path = pathlib.Path(out_path).resolve()
    f = gzip.open(full_out_path, "wb") if out_path.endswith(".jsonl.gz") else full_out_path.open("w")
    writer = jsonlines.Writer(f)
    logger.debug(f"Writing output to {full_out_path}...")
    writer.write_all(examples)
    logger.debug(f"DONE writing")
    f.close()


def filter_datasets(paths, out_path: str, require_fields=[]):
    logger.debug(f"Requiring fields {require_fields}")
    total_lines = 0
    examples = []
    for path in paths:
        full_path = pathlib.Path(path).resolve()
        f = gzip.open(full_path, "rb") if path.endswith(".jsonl.gz") else full_path.open("r")
        reader = jsonlines.Reader(f)

        logger.debug(f"Loading {full_path}")
        for json_dict in tqdm.tqdm(reader, desc=full_path.name):
            total_lines += 1
            # Check all required fields are present
            if any([field not in json_dict or not json_dict[field] for field in require_fields]):
                continue

            # We need the identifier (method name) as a label. Filter invalid identifiers
            if 'identifier' in require_fields and _valid_identifier_regex.match(json_dict['identifier']) == None:
                print(f"WARN: Invalid identifier {require_fields['identifier']}, skipping record")
                continue

            examples.append(json_dict)
            if total_lines % 100000 == 0:
                logger.debug(f"Filtered jsonl to {len(examples)}/{total_lines}")
        f.close()

        logger.debug(f"DONE: Filtered jsonl to {len(examples)}/{total_lines}")

    # TODO: Subsample
    
    # Write output
    full_out_path = pathlib.Path(out_path).resolve()
    f = gzip.open(full_out_path, "wb") if out_path.endswith(".jsonl.gz") else full_out_path.open("w")
    writer = jsonlines.Writer(f)
    logger.debug(f"Writing output to {full_out_path}...")
    writer.write_all(examples)
    logger.debug(f"DONE writing")
    f.close()


if __name__ == "__main__":
    fire.Fire({
        "filter_dataset": filter_dataset,
        "filter_datasets": filter_datasets
    })
