from typing import Iterable

import pandas as pd
from torch.utils.data import Dataset
from transformers.data.datasets.language_modeling import TextDataset
from transformers import BertTokenizerFast

class PretrainingDataset(TextDataset):
    def __init__(self, shuffle=False):
        self.tokenizer = self.make_tokenizer()
        self.data_df = pd.read_parquet('data/codesearchnet_javascript/augmented_data.parquet')
        if shuffle:
            self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)  # shuffle dataset
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, i):
        if not isinstance(i, Iterable):
            i = [i]
        data_batch = [x['text'] for x in self.data_df[i]]
        self.tokenizer.encode_batch(data_batch)

    @staticmethod
    def make_tokenizer(path="huggingface/8k_bpe/8k_bpe-vocab.txt"):
        tok = BertTokenizerFast(path, clean_text=True, lowercase=False, strip_accents=True)
        tok.set_truncation_and_padding(PaddingStrategy.DO_NOT_PAD, max_length=512)
        return tok


if __name__ == "__main__":
    tok = make_tokenizer()
    print(tok.encode("public static void main(String[] args)"))
    print(tok.encode("public static void main(String[] args)").tokens)
