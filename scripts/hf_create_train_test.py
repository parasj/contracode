from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import gzip
from tqdm.auto import tqdm

DATA_PICKLE_PATH = Path('data/codesearchnet_javascript/javascript_augmented.pickle.gz')
CACHE_PATH = Path('data/codesearchnet_javascript/augmented_data.parquet')
TRAIN_OUT_PATH = Path('data/huggingface/pretrain_train.txt')
TEST_OUT_PATH = Path('data/huggingface/pretrain_test.txt')


if __name__ == "__main__":
    if CACHE_PATH.exists():
        df = pd.read_parquet('data/codesearchnet_javascript/augmented_data.parquet')
    else:
        with gzip.open(DATA_PICKLE_PATH) as f:
            data = pickle.load(f)

        flattened_data = []
        for idx, x in enumerate(tqdm(data)):
            for item in x:
                flattened_data.append(dict(data_idx=idx, text=item))

        df = pd.DataFrame(flattened_data)

        del data
        del flattened_data
        df.to_parquet(str(CACHE_PATH.resolve()), engine='pyarrow')

    data_idxs = np.asarray(list(set(df['data_idx'])))
    np.random.shuffle(data_idxs)
    test_idxs, train_idxs = data_idxs[:10000], data_idxs[10000:]
    train_text = df[df['data_idx'].isin(train_idxs)]['text'].sample(frac=1).reset_index(drop=True).tolist()
    test_text = df[df['data_idx'].isin(test_idxs)]['text'].sample(frac=1).reset_index(drop=True).tolist()

    with TRAIN_OUT_PATH.open('w') as f:
        f.write('\n'.join(train_text))

    with TEST_OUT_PATH.open('w') as f:
        f.write('\n'.join(test_text))
