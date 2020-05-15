import re
from collections import Counter


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
        self.camel_case_re = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")

    def camel_case_split(self, identifier):
        """from https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python"""
        return [m.group(0) for m in self.camel_case_re.finditer(identifier)]

    def snake_case_split(self, identifier):
        return identifier.split("_")

    def split_method_name(self, method_name: str):
        toks = [method_name]
        if self.tokenize_snake_case:
            toks = [tok for s in toks for tok in self.snake_case_split(s)]
        if self.tokenize_camel_case:
            toks = [tok for s in toks for tok in self.camel_case_split(s)]
        return toks

    def count_tokens(self, token_list):
        count = Counter()
        for token in token_list:
            if self.case_insensitive:
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
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * float(precision * recall) / float(precision + recall) if (precision + recall) > self.eps else 0
        return precision, recall, f1
