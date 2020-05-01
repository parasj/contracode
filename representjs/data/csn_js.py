import gzip
import jsonlines
import io
import os

import torch
import tqdm


def _make_example(json_dict, fields):
    out_dict = {}
    for json_key, out_key in fields.items():
        out_dict[out_key] = json_dict[json_key]
    return out_dict


class JSONLinesDataset(torch.utils.data.Dataset):
    """Defines a Dataset of columns stored in jsonlines format."""

    def __init__(self, path, fields, **kwargs):
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
        self.examples = [_make_example(line, fields) for line in tqdm.tqdm(reader, f"Loading {path}")]
        f.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


if __name__ == "__main__":
    # Possible keys:
    #   'identifier' for method name which may be blank for anonymous fns
    #   'function' for the function as a string
    #   'function_tokens'
    #   'docstring' for the docstring (blank for most, filled in for about 100k)
    #   'docstring_tokens'
    data_dir = "data/codesearchnet_javascript"
    train_filepath = os.path.join(data_dir, "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl")
    fields = {"function": "function"}
    train_dataset = JSONLinesDataset(train_filepath, fields)
    print("Number of training functions:", len(train_dataset[0]))
    print("Example": train_dataset[0])