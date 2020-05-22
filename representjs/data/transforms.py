from typing import List

import numpy as np
import torch
from loguru import logger
from torchtext.data import load_sp_model

from data.util import Timer, normalize_program
from data.old_dataloader import _augment_server


class Transform:
    def __call__(self, sample):
        raise NotImplementedError()


class NumericalizeTransform(Transform):
    def __init__(self, spm_unigram_path: str, subword_regularization_alpha: float = 0.0, max_length: int = 1024):
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
        normalized_text = normalize_program(sample["function"])
        if self.subword_regularization_alpha:
            # using subword regularization: https://arxiv.org/pdf/1804.10959.pdf
            # NOTE: what is the second argument here (-1)?
            X = self.sp_model.SampleEncodeAsIds(normalized_text, -1, self.subword_regularization_alpha)
        else:
            # using the best decoding
            X = self.sp_model.EncodeAsIds(normalized_text)
        sample["function_ids"] = torch.tensor([self.bos_id] + X[: (self.max_length - 2)] + [self.eos_id])

        if "label" in sample.keys():
            label_ids = self.sp_model.EncodeAsIds(sample["label"])
            sample["label_ids"] = torch.tensor([self.bos_id] + label_ids + [self.eos_id])
        else:
            sample["label_ids"] = None
        return sample


class WindowLineCropTransform(Transform):
    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self, sample):
        assert isinstance(sample, dict) and "function" in sample, "Got bad sample in WindowLineCropTransform: " + str(sample)
        text = sample["function"]
        lines = text.split("\n")  # skip first and last line, usually function signature
        first_idx, last_idx = 1, max(2, len(lines) - 1 - self.window_size)
        window_start = np.random.randint(first_idx, last_idx)
        window_end = min(len(lines), window_start + self.window_size)
        sample["function"] = "\n".join(lines[window_start:window_end])
        return sample


class CanonicalizeKeysTransform(Transform):
    """Clean extra keys from data sample"""

    def __init__(self, **kwargs):
        self.key_mapping = kwargs

    def __call__(self, sample):
        out_dict = {}
        for dest_key, source_key in self.key_mapping.items():
            if source_key not in sample.keys():
                logger.error("Data sample missing key {}, has {}. Destination map was {}.".format(source_key, sample.keys(), dest_key))
            out_dict[dest_key] = sample[source_key]
        return out_dict


class ComposeTransform(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class NodeServerTransform(Transform):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, sample):
        transformed = _augment_server([{"src": sample["function"], "augmentations": self.augmentations}])
        assert len(transformed) == 1
        return {"function": transformed[0]}
