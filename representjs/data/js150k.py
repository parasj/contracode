import json
import logging
import pickle
from json import JSONDecodeError
from typing import Dict

from representjs import PACKAGE_ROOT
from representjs.data.transforms.util import dispatch_to_node

AST_TO_AST_BANK = set([])
SRC_TO_SRC_BANK = set()
VALID_AUGMENTATIONS = AST_TO_AST_BANK.union(SRC_TO_SRC_BANK)


class JS150kDataset:
    def __init__(self):
        logging.info("Loading train dataset")
        self.train_methods = list(self.extract_methods(self.load_dataset('train')))
        logging.info("Loading eval dataset")
        self.eval_methods = list(self.extract_methods(self.load_dataset('eval')))

    @staticmethod
    def extract_methods(dataset: Dict[str, str], dispatch_batch_size=128):
        keys = list(dataset.keys())
        methods = []
        for i in range(0, len(keys), dispatch_batch_size):
            dispatch_data = [{"src": dataset[key], "augmentations": [{"fn": "extract_methods"}]} for key in keys[i: i + dispatch_batch_size]]
            json_dispatch = json.dumps(dispatch_data)
            stdout, stderr = dispatch_to_node('transform.js', json_dispatch)
            try:
                method_data = json.loads(stdout)
            except JSONDecodeError as e:
                logging.exception(e)
                logging.error(dispatch_data)
                logging.error("Got stdout: {}".format(stdout))
                logging.error("Got stderr: {}".format(stderr))
            print(method_data)
            return
        return methods

    @staticmethod
    def load_dataset(config):
        """Generator that loads a pickled dataset of js150k and extracts all methods"""
        path = PACKAGE_ROOT / "data" / "js150k_{}.pkl".format(config)
        with path.open('rb') as f:
            data = pickle.load(f)
        data_out = dict()
        for tup in data:
            path = tup['path']
            js_src = tup['js']
            data_out[path] = js_src
        return data_out


if __name__ == "__main__":
    ds = JS150kDataset()
    print(len(ds.train_methods))
    print(len(ds.train_methods))
