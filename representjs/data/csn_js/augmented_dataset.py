import pprint

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data.transforms import Transform, WindowLineCropTransform, CanonicalizeKeysTransform, ComposeTransform
from representjs import CSNJS_DIR
from data.csn_js.jsonl_dataset import JSONLinesDataset, get_csnjs_dataset
from representjs.pretrain import DEFAULT_CSNJS_TRAIN_FILEPATH


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
            key = self.transform(sample.copy())
            query = self.transform(sample.copy())
            assert 'data' in key.keys() and 'data' in query.keys()
            out_dict = {'data_key': key['data'], 'data_query': query['data']}
            return out_dict
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            return sample


class PadCollateWrapper:
    """Object enables pickle versus lambda"""
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
    train_dataset = get_csnjs_dataset(DEFAULT_CSNJS_TRAIN_FILEPATH, label_mode="none", limit_size=100)
    test_transforms = ComposeTransform([
        WindowLineCropTransform(6),
        # NumericalizeTransform(SPM_UNIGRAM_FILEPATH, 0., 1024),
        CanonicalizeKeysTransform(data='function'),
    ])
    augmented_dataset = AugmentedJSDataset(train_dataset, test_transforms, contrastive=True)
    for i, data in enumerate(augmented_dataset[0:10]):
        logger.info("Program {}".format(i))
        logger.info("Got data object: \n" + pprint.pformat(data))
        print()
        print()
