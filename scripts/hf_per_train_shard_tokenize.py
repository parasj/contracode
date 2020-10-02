import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from transformers import BertTokenizerFast
from tqdm import tqdm

if __name__ == "__main__":
    assert len(sys.argv) == 2
    data_shard_idx = int(sys.argv[1])
    data_shard_path = f"/data/ajay/contracode/data/hf_data/train_chunks/augmented_pretrain_df.{data_shard_idx:04d}.train.pickle.gz"
    data_shard_path_out = (
        f"/data/ajay/contracode/data/hf_data/train_chunks_tokenized/augmented_pretrain_tokenized_df.{data_shard_idx:04d}.train.pickle.gz"
    )

    def load_tokenizer(path="data/vocab/8k_bpe/8k_bpe-vocab.txt"):
        return BertTokenizerFast(path, clean_text=True, lowercase=False, strip_accents=True, unk_token="<unk>")

    def load_data(path):
        return pd.read_pickle(path)

    tokenizer = load_tokenizer()
    df_shard = load_data(data_shard_path)
    tqdm.pandas()
    df_shard["toks"] = df_shard["text"].progress_apply(lambda x: np.asarray(tokenizer.encode(x)))
    df_shard = df_shard[["data_idx", "toks"]]

    df_shard.to_pickle(data_shard_path_out)
