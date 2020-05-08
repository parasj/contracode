import pprint
from typing import List

import numpy as np
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.data import load_sp_model

from representjs import CSNJS_DIR
from representjs.data.csn_js_jsonl import JSONLinesDataset, get_csnjs_dataset
from representjs.data.csn_js_loader import normalize_program
from representjs.data.util import Timer
from representjs.pretrain import CSNJS_TRAIN_FILEPATH


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
        assert isinstance(sample, dict) and 'function' in sample, "Got bad sample in WindowLineCropTransform: " + str(sample)
        text = sample['function']
        lines = text.split('\n')  # skip first and last line, usually function signature    
        first_idx, last_idx = 1, max(2, len(lines) - 1 - self.window_size)
        window_start = np.random.randint(first_idx, last_idx)
        window_end = min(len(lines), window_start + self.window_size)
        sample['function'] = "\n".join(lines[window_start:window_end])
        return sample


class CanonicalizeKeysTransform(Transform):
    """Clean extra keys from data sample"""

    def __init__(self, **kwargs):
        self.key_mapping = kwargs

    def __call__(self, sample):
        out_dict = {}
        for dest_key, source_key in self.key_mapping.items():
            if source_key not in sample.keys():
                logger.error(
                    "Data sample missing key {}, has {}. Destination map was {}.".format(source_key, sample.keys(), dest_key))
            out_dict[dest_key] = sample[source_key]
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
        samples = self.json_dataset[idx]
        if isinstance(samples, list):
            return [self.augment_element(sample) for sample in samples]
        else:
            return self.augment_element(samples)

    def augment_element(self, sample):
        if self.contrastive:
            assert self.transform is not None, "Must specify a transformation if creating contrastive dataset"
            key = self.transform(sample)
            query = self.transform(sample)
            assert 'data' in key.keys() and 'data' in query.keys()
            out_dict = {'data_key': key['data'], 'data_query': query['data']}
            return out_dict
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            return sample


class PadCollateWrapper:
    def __init__(self, contrastive=False, pad_id=None, batch_first=True):
        self.batch_first = batch_first
        self.contrastive = contrastive
        self.pad_id = pad_id

    def __call__(self, batch):
        batch_size = len(batch)
        if self.contrastive:
            assert 'data_key' in batch[0].keys() and 'data_query' in batch[0].keys(), "Missing contrastive keys, {}".format(
                batch[0].keys())
            data_key_list = [sample['data_key'] for sample in batch]
            data_query_list = [sample['data_query'] for sample in batch]
            data = pad_sequence(data_key_list + data_query_list, padding_value=self.pad_id, batch_first=True)  # [2B, T]
            assert data.size(0) == 2 * batch_size
            data = data.view(2, batch_size, data.size(-1)).transpose(0, 1).contiguous()
            return data, None
        else:
            data_list = [sample['data'] for sample in batch]
            label_list = [sample['label'] for sample in batch]
            data = pad_sequence(data_list, padding_value=self.pad_id, batch_first=self.batch_first)
            label = pad_sequence(label_list, padding_value=self.pad_id, batch_first=self.batch_first)
            return data, label


if __name__ == "__main__":
    SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")
    train_dataset = get_csnjs_dataset(CSNJS_TRAIN_FILEPATH, label_mode="none", limit_size=100)
    test_transforms = ComposeTransform([
        WindowLineCropTransform(6),
        # NumericalizeTransform(SPM_UNIGRAM_FILEPATH, 0., 1024),
        # CanonicalizeKeysTransform(data='function_ids'),
    ])
    augmented_dataset = AugmentedJSDataset(train_dataset, test_transforms)
    for i, data in enumerate(augmented_dataset[0:10]):
        logger.info("Program {}".format(i))
        logger.info("Got data object: \n" + pprint.pformat(data))
        logger.debug(data['function'])
