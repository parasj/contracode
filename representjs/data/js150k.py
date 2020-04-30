import pickle
from typing import List

import representjs

AST_TO_AST_BANK = set([])
SRC_TO_SRC_BANK = set()
VALID_AUGMENTATIONS = AST_TO_AST_BANK.union(SRC_TO_SRC_BANK)


class JS150kDataset:
    def __init__(self):
        self.train_methods = dict(self.extract_methods(self.load_dataset('train')))
        self.eval_methods = dict(self.extract_methods(self.load_dataset('eval')))

    @staticmethod
    def load_dataset(config):
        """Generator that loads a pickled dataset of js150k and extracts all methods"""
        path = representjs.PACKAGE_ROOT / "data" / "js150k_{}.pkl".format(config)
        with path.open('rb') as f:
            data = pickle.load(f)
        data_out = dict()
        for tup in data:
            path = tup['path']
            js_src = tup['js']
            data_out[path] = js_src

if __name__ == "__main__":
    ds = JS150kDataset()
    print(len(ds.train_methods))
    print(len(ds.train_methods))
