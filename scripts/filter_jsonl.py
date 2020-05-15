import gzip
import json
import jsonlines
import re

import pathlib
import fire
import tqdm
from loguru import logger

from data.jsonl_dataset import _fix_json_dict
from data.util import dispatch_to_node

_valid_identifier_regex = re.compile(r"^[a-zA-Z_$][0-9a-zA-Z_$]*$")


def has_transform_error(json_dict):
    # Try to parse/transform the code, and filter out if we can't
    # Set up transformation input
    json_dict = _fix_json_dict(json_dict, ["function"], "function", "identifier")

    transform_payload = [
        dict(src=json_dict["function"], augmentations=[{"fn": "identity_ast2ast"}],)  # TODO: this key should be "code" for supervised set
    ]
    transform_payload = json.dumps(transform_payload)
    stdout, stderr = dispatch_to_node("transform.js", transform_payload)
    if stderr:
        return True, None

    return False, json_dict


from multiprocessing import Pool


def filter_dataset_parallel(path: str, out_path: str):
    full_path = pathlib.Path(path).resolve()
    read_f = gzip.open(full_path, "rb") if path.endswith(".jsonl.gz") else full_path.open("r")
    reader = jsonlines.Reader(read_f)

    full_out_path = pathlib.Path(out_path).resolve()
    write_f = gzip.open(full_out_path, "wb") if out_path.endswith(".jsonl.gz") else full_out_path.open("w")
    writer = jsonlines.Writer(write_f)
    logger.debug(f"Writing output to {full_out_path}...")

    examples = []
    logger.debug(f"Loading {full_path}")
    pool = Pool(processes=16)
    total_lines = 0
    num_written = 0
    for has_error, json_dict in tqdm.tqdm(pool.imap(has_transform_error, reader, chunksize=4), desc=full_path.name):
        total_lines += 1
        if not has_error:
            writer.write(json_dict)
            num_written += 1

        if total_lines % 1000 == 0:
            logger.debug(f"Filtered jsonl to {num_written}/{total_lines}")
    logger.debug(f"DONE: Filtered jsonl to {num_written}/{total_lines}")
    read_f.close()
    write_f.close()


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
        if "identifier" in require_fields and _valid_identifier_regex.match(json_dict["identifier"]) == None:
            continue

        # Try to parse/transform the code, and filter out if we can't
        if exclude_transform_errors:
            # Set up transformation input
            transform_payload = [
                dict(
                    src=json_dict["function"],  # TODO: this key should be "code" for supervised set
                    augmentations=[{"fn": "identity_ast2ast"}],
                )
            ]
            transform_payload = json.dumps(transform_payload)
            stdout, stderr = dispatch_to_node("transform.js", transform_payload)
            if stderr:
                continue

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
            if "identifier" in require_fields and _valid_identifier_regex.match(json_dict["identifier"]) == None:
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
    fire.Fire(
        {"filter_dataset": filter_dataset, "filter_datasets": filter_datasets, "filter_dataset_parallel": filter_dataset_parallel,}
    )
