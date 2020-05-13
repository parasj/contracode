import re
from collections import Counter

from data.jsonl_dataset import split_method_name


def gen_counter_items(counts: Counter):
    for key in counts.keys():
        for i in range(counts[key]):
            yield key


class F1MetricMethodName:
    def __init__(self, case_insensitive=True, ignore_empty=True, tokenize_camel_case=True, tokenize_snake_case=True, eps=1e-6):
        self.ignore_empty = ignore_empty
        self.case_insensitive = case_insensitive
        self.tokenize_camel_case = tokenize_camel_case
        self.tokenize_snake_case = tokenize_snake_case
        self.eps = eps

    def split_method_name(self, method_name: str):
        return split_method_name(
            method_name, self.tokenize_snake_case, self.tokenize_camel_case, self.case_insensitive)

    def count_tokens(self, token_list):
        count = Counter()
        for token in token_list:
            if self.case_insensitive:
                # NOTE: redundant, also in split_method_name
                token = token.lower()
            if not (self.ignore_empty and len(token) == 0):
                count[token] += 1
        return count

    def __call__(self, identifier_pred, identifier_target):
        counts_pred = self.count_tokens(self.split_method_name(identifier_pred))
        counts_target = self.count_tokens(self.split_method_name(identifier_target))
        tp, fp, fn = 0, 0, 0
        for token in gen_counter_items(counts_pred):
            if counts_pred[token] > 0 and counts_target[token] > 0:
                tp += 1
                counts_pred[token] -= 1
                counts_target[token] -= 1
            elif counts_pred[token] > 0 and counts_target[token] == 0:
                fp += 1
                counts_pred[token] -= 1
        fn += sum(counts_target.values())
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.
        f1 = 2. * float(precision + recall) / float(precision + recall) if (precision + recall) > self.eps else 0
        return precision, recall, f1
