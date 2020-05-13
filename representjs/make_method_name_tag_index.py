from collections import Counter
import gzip
import json
import jsonlines
from operator import itemgetter
import pickle
import re

import pathlib
import fire
import tqdm
from loguru import logger

from data.jsonl_dataset import _fix_json_dict
from metrics.f1 import F1MetricMethodName


def method_tokens(dataset_paths, output_path):
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    metric = F1MetricMethodName()

    all_method_names = Counter()
    all_toks = Counter()

    for dataset_path in dataset_paths:
        if dataset_path.endswith(".gz"):
            f = gzip.open(dataset_path, "rb")
        else:
            f = open(dataset_path, "r")
        reader = jsonlines.Reader(f)
        method_names = map(lambda d: d["func_name"] if "func_name" in d else d["identifier"], reader)
        method_names = list(tqdm.tqdm(filter(len, method_names), desc=f"Processing {dataset_path}"))
        f.close()

        # Update method name counts
        all_method_names.update(method_names)
        for name in method_names:
            toks = metric.split_method_name(name)
            toks = map(lambda s: s.lower(), toks)
            all_toks.update(toks)
    
    print(all_toks.most_common(500))

    with open(output_path, "wb") as out_f:
        obj = {
            "token_top10": list(map(itemgetter(0), all_toks.most_common(10))),
            "token_top50": list(map(itemgetter(0), all_toks.most_common(50))),
            "token_top100": list(map(itemgetter(0), all_toks.most_common(100))),
            "token_top500": list(map(itemgetter(0), all_toks.most_common(500))),
            "token_top1000": list(map(itemgetter(0), all_toks.most_common(1000))),
            "token_counter": all_toks,
            "method_name_counter": all_method_names,
        }
        pickle.dump(obj, out_f, protocol=pickle.HIGHEST_PROTOCOL)

# TODO: Inspect try catch blocks - can we predict exception type?


if __name__=="__main__":
    fire.Fire(method_tokens)