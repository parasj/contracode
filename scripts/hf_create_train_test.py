from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import gzip
from tqdm.auto import tqdm

DATA_PICKLE_PATH = Path("data/codesearchnet_javascript/javascript_augmented.pickle.gz")
CACHE_PATH = Path("data/hf_data/augmented_pretrain_data_df.parquet")
TRAIN_OUT_PATH = Path("/data/paras/augmented_pretrain_df.train.pickle.gz")
TEST_OUT_PATH = Path("/data/paras/augmented_pretrain_df.test.pickle.gz")
TRAIN_OUT_TXT_PATH = Path("/data/paras/augmented_pretrain_df.train.txt")
TEST_OUT_TXT_PATH = Path("/data/paras/augmented_pretrain_df.test.txt")


if __name__ == "__main__":
    if CACHE_PATH.exists():
        print("Loading from cache")
        df = pd.read_parquet(CACHE_PATH)
    else:
        print("Loading from pickle")
        with gzip.open(DATA_PICKLE_PATH) as f:
            data = pickle.load(f)

        flattened_data = []
        for idx, x in enumerate(tqdm(data)):
            for item in x:
                flattened_data.append(dict(data_idx=idx, text=item))
        df = pd.DataFrame(flattened_data)
        del data, flattened_data

        print("Saving cache file of dataframe")
        df.to_parquet(str(CACHE_PATH.resolve()), engine="pyarrow")

    data_idxs = np.asarray(list(set(df["data_idx"])))
    np.random.shuffle(data_idxs)
    test_idxs, train_idxs = data_idxs[:10000], data_idxs[10000:]
    train_df = df[df["data_idx"].isin(train_idxs)].sample(frac=1).reset_index(drop=True)
    test_df = df[df["data_idx"].isin(test_idxs)].sample(frac=1).reset_index(drop=True)

    print("Saving train data")
    train_df.to_pickle(TRAIN_OUT_PATH)

    print("Saving test data")
    test_df.to_pickle(TEST_OUT_PATH)

    train_txt = train_df["text"].tolist()
    test_txt = test_df["text"].tolist()

    print("Saving train text")
    with TRAIN_OUT_TXT_PATH.open("w") as f:
        f.write("\n".join(train_txt))
    print("Saving test text")
    with TEST_OUT_TXT_PATH.open("w") as f:
        f.write("\n".join(test_txt))
