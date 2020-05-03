import gzip
import json
import jsonlines
import os
from typing import List

import torch
import tqdm

from data.util import dispatch_to_node

# Possible keys:
#   'identifier' for method name which may be blank for anonymous fns
#   'function' for the function as a string
#   'function_tokens'
#   'docstring' for the docstring (blank for most, filled in for about 100k)
#   'docstring_tokens'
FUNCTION_ONLY_FIELDS = {"function": "function"}


def _make_example(json_dict, fields):
    out_dict = {}
    for json_key, out_key in fields.items():
        out_dict[out_key] = json_dict[json_key]
    return out_dict


class JSONLinesDataset(torch.utils.data.Dataset):
    """Defines a Dataset of columns stored in jsonlines format."""

    def __init__(self, path, fields=FUNCTION_ONLY_FIELDS, **kwargs):
        """Create a JSONLinesDataset given a path and field mapping dictionary.

        Arguments:
            path (str): Path to the data file. Must be in .jsonl.gz or .jsonl format.
            fields (dict[str: str]:
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


def javascript_augmentation_collate_fn(examples: List[dict], augmentations: List[dict]):
    """Augments and batches a list of function dicts.

    Arguments:
        examples (List[dict[str, Any]]). The dicts must have key "function".
        augmentations (List[dict]). Augmentations to apply to the functions.
            example: [{"fn": "extract_methods"}]
    """
    B = len(examples)

    # Set up transformation input, duplicated for 2 agumentations per function
    transform_payload = []
    for _ in range(2):
        for example in examples:
            transform_payload.append(dict(
                src=example["function"],
                augmentations=augmentations))
    transform_payload = json.dumps(transform_payload)

    # Transform code
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

    t1, t2 = transformed[:B], transformed[B:]
    print("t1", t1)
    print("t2", t2)

def javascript_augmentation_dataloader(dataset, augmentations: List[dict], **kwargs):
    collate_fn = lambda batch: javascript_augmentation_collate_fn(batch, augmentations)
    return torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **kwargs)


if __name__ == "__main__":
    data_dir = "data/codesearchnet_javascript"
    train_filepath = os.path.join(data_dir, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl")
    train_dataset = JSONLinesDataset(train_filepath)
    print("Number of training functions:", len(train_dataset[0]))
    print("Example", train_dataset[0])

    augmentations = [{"fn": "insert_noop"}]  # TODO: Pass probability of applying each transform
    train_loader = javascript_augmentation_dataloader(
        train_dataset, batch_size=2, shuffle=False, augmentations=augmentations)
    for batch in train_loader:
        import IPython
        IPython.embed()
        break
