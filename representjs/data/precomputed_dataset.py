import pathlib
import pickle
import re
from typing import Optional, Iterable

import numpy as np
import torch
import tqdm
from loguru import logger

from data.util import normalize_program


class PrecomputedDataset(torch.utils.data.Dataset):
    """Defines a Dataset of unsupervised programs stored in pickle format."""

    def __init__(self,
                 path,
                 sp,
                 min_alternatives=1,
                 limit_size=-1,
                 max_length=1024,
                 subword_regularization_alpha=0.1,
                 program_mode="identity"):
        """Create a JSONLinesDataset given a path and field mapping dictionary.
        Arguments:
            path (str): Path to the data file. Must be in .pickle format.
        """
        super().__init__()
        full_path = pathlib.Path(path).resolve()
        logger.debug(f"Loading {full_path}")
        with full_path.open("rb") as f:
            self.examples = pickle.load(f)
        logger.debug(f"Loaded {len(self.examples)} examples")
        if limit_size > 0:
            self.examples = self.examples[:limit_size]
            logger.debug(f"Limited size: took first {limit_size} examples")
        self.examples = list(map(list, self.examples))
        logger.debug(f"Converted examples to lists of alternatives")
        if min_alternatives:
            self.examples = list(filter(lambda ex: len(ex) >= min_alternatives, self.examples))
        logger.debug(f"Filtered dataset to {len(self.examples)} examples with at least {min_alternatives} alternatives")

        self.program_mode = program_mode
        self.max_length = max_length
        self.subword_regularization_alpha = subword_regularization_alpha
        self.sp = sp
        self.bos_id = sp.PieceToId("<s>")
        self.eos_id = sp.PieceToId("</s>")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        alternatives = self.examples[idx]
        n_alt = len(alternatives)
        if self.program_mode == "identity":
            return self.encode(alternatives[0])
        elif self.program_mode == "augmentation":
            i = np.random.randint(n_alt)
            return self.encode(alternatives[i])
        elif self.program_mode == "contrastive":
            i = np.random.randint(n_alt)
            j = i
            if n_alt > 1:
                while j == i:
                    j = np.random.randint(n_alt)
            return (self.encode(alternatives[i]), self.encode(alternatives[j]))
        else:
            raise ValueError(f"Invalid program mode {self.program_mode}")

    def encode(self, program):
        program = normalize_program(program)

        # Encode as ids with sentencepiece
        if self.subword_regularization_alpha:
            # using subword regularization: https://arxiv.org/pdf/1804.10959.pdf
            # NOTE: what is the second argument here (-1)?
            program = self.sp.SampleEncodeAsIds(program, -1, self.subword_regularization_alpha)
        else:
            # using the best decoding
            program = self.sp.EncodeAsIds(program)
        
        return torch.tensor([self.bos_id] + program[:(self.max_length - 2)] + [self.eos_id])
