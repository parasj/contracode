import pickle
from pathlib import Path
from typing import Dict, List

import representjs

JS150K_RAW_DATA_TYPE = List[Dict]

class JS150kDataset:
    def __init__(self):
        self._eval_data_obj = None
        self._train_data_obj = None

    @staticmethod
    def _load_dataset(dataset_path: Path):
        with dataset_path.open('rb') as f:
            return pickle.load(f)

    def eval_data_raw(self) -> JS150K_RAW_DATA_TYPE:
        if self._eval_data_obj is None:
            self._eval_data_obj = self._load_dataset(representjs.PACKAGE_ROOT / "data" / "js150k_eval.pkl")
        return self._eval_data_obj

    def train_data_raw(self) -> JS150K_RAW_DATA_TYPE:
        if self._eval_data_obj is None:
            self._eval_data_obj = self._load_dataset(representjs.PACKAGE_ROOT / "data" / "js150k_eval.pkl")
        return self._eval_data_obj
