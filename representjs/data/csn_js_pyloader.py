import random
from typing import List

import torch
from loguru import logger
from torch.utils.data import Dataset
from torchtext.data import load_sp_model

from data.csn_js_jsonl import JSONLinesDataset
from data.csn_js_loader import normalize_program
from data.util import Timer


class Transform:
    def __call__(self, sample):
        raise NotImplementedError()


class NumericalizeTransform(Transform):
    def __init__(self, spm_unigram_path: str, subword_regularization_alpha: float = 0., max_length: int = 1024):
        self.max_length = max_length
        self.subword_regularization_alpha = subword_regularization_alpha
        self.spm_unigram_path = spm_unigram_path
        self.sp_model = load_sp_model(spm_unigram_path)
        self.bos_id = self.sp_model.PieceToId("<s>")
        self.eos_id = self.sp_model.PieceToId("</s>")
        self.pad_id = self.sp_model.PieceToId("[PAD]")

    def __getstate__(self):
        return self.spm_unigram_path, self.subword_regularization_alpha, self.max_length

    def __setstate__(self, state):
        with Timer() as t:
            (self.spm_unigram_path, self.subword_regularization_alpha, self.max_length) = state
            self.sp_model = load_sp_model(self.spm_unigram_path)
            self.bos_id = self.sp_model.PieceToId("<s>")
            self.eos_id = self.sp_model.PieceToId("</s>")
            self.pad_id = self.sp_model.PieceToId("[PAD]")
        logger.info("Hydrating vocabulary took {:.3f}s".format(t.interval))

    def __call__(self, sample):
        normalized_text = normalize_program(sample['function'])
        if self.subword_regularization_alpha:
            # using subword regularization: https://arxiv.org/pdf/1804.10959.pdf
            # NOTE: what is the second argument here (-1)?
            X = self.sp_model.SampleEncodeAsIds(normalized_text, -1, self.subword_regularization_alpha)
        else:
            # using the best decoding
            X = self.sp_model.EncodeAsIds(normalized_text)
        sample['function_ids'] = torch.tensor([self.bos_id] + X[:(self.max_length - 2)] + [self.eos_id])

        if "label" in sample.keys():
            label_ids = self.sp_model.EncodeAsIds(sample["label"])
            sample['label_ids'] = torch.tensor([self.bos_id] + label_ids + [self.eos_id])
        else:
            sample['label_ids'] = None
        return sample


class WindowLineCropTransform(Transform):
    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self, sample):
        text = sample['function']
        lines = text.split('\n')  # skip first and last line, usually function signature
        first_idx, last_idx = 1, len(lines) - 2
        window_start = random.randint(first_idx, last_idx - self.window_size)
        sample['function'] = "\n".join(lines[window_start, window_start + self.window_size])
        return sample


class CleanKeysTransform(Transform):
    """Clean extra keys from data sample"""
    def __init__(self, data_key: str, label_key: str = None):
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, sample):
        out_dict = {'data': sample[self.data_key]}
        if self.label_key is not None:
            out_dict['label'] = sample[self.label_key]
        return out_dict


class ComposeTransform(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class AugmentedJSDataset(Dataset):
    def __init__(self, json_dataset: JSONLinesDataset, transform: Transform = None, max_length: int = 1024, contrastive=False):
        self.contrastive = contrastive
        self.json_dataset = json_dataset
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.json_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.json_dataset[idx]
        if self.contrastive:
            assert self.transform is not None, "Must specify a transformation if creating contrastive dataset"
            key = self.transform(sample)
            query = self.transform(sample)
            assert 'data' in key.keys() and 'data' in query.keys()
            out_dict = {'data_key': key['data'], 'data_query': query['data']}
            if 'label' in key.keys():
                assert 'label' in query.keys() and query['label'] == key['label']
                out_dict['label'] = key['label']
            return out_dict
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            return sample
