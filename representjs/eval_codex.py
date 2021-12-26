import json
import os
import time
import random

import fire
from loguru import logger
import numpy as np
import openai
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
import torch
import wandb

from data.deeptyper_dataset import _tokenize, load_type_vocab 
from data.jsonl_dataset import get_csnjs_dataset
from data.old_dataloader import javascript_dataloader
from data.util import Timer
from metrics.f1 import F1MetricMethodName


# Default argument values
DATA_DIR = "data/codesearchnet_javascript"
CSNJS_TEST_FILEPATH = os.path.join(DATA_DIR, "javascript_test_0.jsonl.gz")
TYPE_VOCAB_FILEPATH = os.path.join("data", "type_prediction", "target_wl")
TYPE_EVAL_DATA_FILEPATH = os.path.join("data", "type_prediction", "valid_nounk.txt")
SPM_UNIGRAM_FILEPATH = os.path.join(DATA_DIR, "csnjs_8k_9995p_unigram_url.model")
torch.manual_seed(0)


class CodexAPI:
    def __init__(self, api_key, min_interval=.1):
        self.api_key = api_key
        openai.api_key = api_key
        self.last_req_time = time.time()
        self.min_interval = min_interval

    def pause(self):
        time.sleep(max(0, self.last_req_time + self.min_interval - time.time()))
        self.last_req_time = time.time()


    def get_method_names(self, code, engine='davinci-codex', **kwargs):
        def get_completion_block(code, gt=None):
            prompt = "// anonymous function definition\n"
            prompt += code
            prompt += "\n\n// alternative name for above function is"
            if gt:
                prompt += " " + gt
            return prompt

        # prompt = get_completion_block("function x(a, b) {\nreturn a + b;\n}", gt='concatenate')
        # prompt += "\n\n-------------\n\n"
        prompt = get_completion_block(code, gt=None)

        args = dict(
            temperature=0.0,
            max_tokens=10,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        args.update(kwargs)
        self.pause()
        results = openai.Completion.create(
            prompt=prompt,
            engine=engine,
            stop=[";", "\n"],
            n=1,
            **args,
        )
        s = results['choices'][0]['text'].strip()
        return ''.join([i for i in s if i.isalpha()])

    def get_types(self, typescript_code_tokens, types_per_token, engine='davinci-codex', truncate_len=1536, n_shot=0, **kwargs):
        header = "// Task: annotate types for a Typescript program. Use any if type is unknown.\n\n"
        sample_a = """// Typescript source
export const customVariables = { type : 's' , UNK0 : { "s" : { type : 's' , UNK1 : { "s" : { type : 's' } } } } } ;

// Types of variables and identifiers
// Vars: [customVariables, type, UNK0, type, UNK1, type]
// Types: [complex, string, complex, string, complex, string]
"""

        sample_b = """// Typescript source
Polymer ( { anchor : 's' , UNK0 : function ( ) { this . fire ( 's' , { name : 's' , data : { anchor : this . anchor } } ) ; } , } ) ;

// Types of variables and identifiers
// Vars: [Polymer, anchor, UNK0, fire, name, data, anchor, anchor]
// Types: [void, string, void, any, string, complex, any, any]
"""

        def get_completion_block(code, var_names, gt_type_list=None):
            prompt = "// Typescript source\n"
            prompt += code
            prompt += "\n\n// Types of variables and identifiers\n"
            prompt += "// Vars: [" + ", ".join(var_names) + "]\n"
            prompt += "// Types: ["
            if gt_type_list:
                prompt += ", ".join(gt_type_list) + "]"
            return prompt

        def replace_unk(code, types_per_token, unk_token="_UNKNOWN_", tokens_to_drop=("<s>", "</s>")):
            out_tokens = []
            out_var_tokens = []
            out_var_types = []
            unk_idx = 0
            for tok, typ in zip(code, types_per_token):
                if tok in tokens_to_drop:
                    continue
                elif typ is None or typ == '':
                    out_tokens.append(tok)
                elif tok == unk_token:
                    out_tokens.append(f"UNK{unk_idx}")
                    out_var_tokens.append(f"UNK{unk_idx}")
                    out_var_types.append(typ)
                    unk_idx += 1
                else:
                    out_tokens.append(tok)
                    out_var_tokens.append(tok)  
                    out_var_types.append(typ)
            return out_tokens, out_var_tokens, out_var_types

        if len(typescript_code_tokens) > truncate_len:
            logger.warning(f"Truncating to {truncate_len} tokens from {len(typescript_code_tokens)}")
        typescript_code_tokens = typescript_code_tokens[:truncate_len]
        types_per_token = types_per_token[:truncate_len]

        out_tokens, out_var_tokens, out_var_types = replace_unk(typescript_code_tokens, types_per_token)
        prompt = header
        if n_shot >= 1:
            prompt += "\n" + sample_a + "\n"
        if n_shot >= 2:
            prompt += "\n" + sample_b + "\n"
        prompt += get_completion_block(" ".join(out_tokens), out_var_tokens, gt_type_list=None)

        args = dict(
            temperature=0.0,
            max_tokens=max(1, len(out_var_tokens) * 2),
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        args.update(kwargs)
        self.pause()
        try:
            results = openai.Completion.create(
                prompt=prompt,
                engine=engine,
                stop=["]", ";", "\n"],
                n=1,
                **args,
            )
            pred_var_names = [x.strip() for x in results['choices'][0]['text'].split(',')][:len(out_var_types)]
            pred_var_names += ['any'] * (len(out_var_types) - len(pred_var_names))
            return pred_var_names, out_var_types
        except Exception as e:
            logger.error(f"Retrying with truncate_len = {truncate_len // 2}")
            return self.get_types(typescript_code_tokens, types_per_token, engine=engine, truncate_len=truncate_len // 2, **kwargs)

class DeepTyperDatasetText(torch.utils.data.Dataset):
    """Similar to DeepTyperDataset but does not tokenize input code"""
    def __init__(
        self,
        data_path,
        split_source_targets_by_tab=False,
    ):
        self.split_source_targets_by_tab = split_source_targets_by_tab
        self.lines = []
        with open(data_path, "r") as f:
            for line in f:
                self.lines.append(line.rstrip())
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        deeptyper_line = self.lines[idx]
        if self.split_source_targets_by_tab:
            # Code tokens and type labels are delimeted by tab, as in .json files
            js_tokens, labels = deeptyper_line.split("\t")
        else:
            # Code tokens and type labels are delimited by space after </s>, as in .txt files
            js_end = deeptyper_line.index("</s>") + len("</s>")
            js_tokens = deeptyper_line[:js_end]
            labels = deeptyper_line[js_end + 1 :]
    
        # Split code by spaces to get DeepTyper tokens, excluding <s>, </s>
        js_tokens = js_tokens.split(" ")[1:-1]
        labels = labels.split(" ")[1:-1]
        labels = [l.strip("$") for l in labels]
        labels = ['' if x == "O" else x for x in labels]
        assert len(js_tokens) == len(labels)
        return js_tokens, labels


def method_naming(
    api_key: str,
    prediction_save_path: str = None,
    prediction_save_every_n_iters: int = 16,
    engine: str = 'davinci-codex',
    gpt_temperature: float = 0.0,
    gpt_top_p: float = 1.0,
    gpt_frequency_penalty: float = 0.0,
    gpt_presence_penalty: float = 0.0,
    gpt_max_tokens: int = 10,
    test_filepath: str = CSNJS_TEST_FILEPATH,
    spm_filepath: str = SPM_UNIGRAM_FILEPATH,
    limit_dataset_size=-1,
    program_mode="identity",
    label_mode="identifier",
):
    config = locals()
    del config['api_key']
    wandb.init(name=f"openai-{engine}", config=config, project="f1_eval", entity="ml4code")
    if prediction_save_path is None:
        prediction_save_path = wandb.run.dir + "/predictions.json"
    logger.info(f"Saving predictions to {prediction_save_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)

    # Create test dataset and dataloader
    logger.info(f"Test data path {test_filepath}")
    test_dataset = get_csnjs_dataset(test_filepath, label_mode=label_mode, limit_size=limit_dataset_size)
    logger.info(f"Test dataset size: {len(test_filepath)}")
    test_loader = javascript_dataloader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        program_mode=program_mode,
        subword_regularization_alpha=0,
        augmentations=[],
        sp=None,
        spm_unigram_path=spm_filepath,
    )
    pad_id = sp.PieceToId("[PAD]")

    # Create codex API
    api = CodexAPI(api_key)

    # Make metric
    metric = F1MetricMethodName()

    # Evaluate
    sample_generations = []
    n_examples = 0
    precision, recall, f1 = 0.0, 0.0, 0.0
    precision_avg, recall_avg, f1_overall = 0.0, 0.0, 0.0
    with tqdm(test_loader, desc="test") as pbar:
        for X, Y, Z, ZZ in pbar:
            with Timer() as t:
                code = sp.Decode(X[0].numpy().tolist()).replace("[EOL]", "\n")
                pred = api.get_method_names(
                    code, engine=engine, temperature=gpt_temperature, top_p=gpt_top_p,
                    frequency_penalty=gpt_frequency_penalty, presence_penalty=gpt_presence_penalty,
                    max_tokens=gpt_max_tokens)
                sample_generations.append(pred)
                # tqdm.write("Code:\n", "  -->  \t" + code.replace('\n', '\n  -->  '), "\n")
                # tqdm.write(f"{pred:16} vs {sp.Decode(Y[0].numpy().tolist()):16} (GT)")
                sample_generations.append(dict(code=code, pred=pred, gt=sp.Decode(Y[0].numpy().tolist())))
            precision_item, score_item, f1_item = metric(pred, sp.Decode(Y[0].numpy().tolist()))
            n_examples += 1
            precision += precision_item
            recall += score_item
            f1 += f1_item

            precision_avg = precision / n_examples
            recall_avg = recall / n_examples
            if precision_avg or recall_avg:
                f1_overall = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
            else:
                f1_overall = 0.0
            item_metrics = {"precision_item": precision_item, "recall_item": score_item, "f1_item": f1_item}
            avg_metrics = {
                "precision_avg": precision_avg,
                "recall_avg": recall_avg,
                "f1_avg": f1 / n_examples,
                "f1_overall": f1_overall,
            }
            pbar.set_description(f"Current F1 = {f1_overall*100.:0.2f}%")
            wandb.log(avg_metrics, step=n_examples)
            wandb.log(item_metrics, step=n_examples)

            if n_examples % prediction_save_every_n_iters == 0:
                with open(prediction_save_path, 'w') as f:
                    json.dump(sample_generations, f)
                wandb.save(prediction_save_path)


    precision_avg = precision / n_examples
    recall_avg = recall / n_examples
    f1_overall = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
    logger.info(f"Precision: {precision_avg*100:.2f}%")
    logger.info(f"Recall: {recall_avg*100:.2f}%")
    logger.info(f"F1: {f1_overall*100:.2f}%")

    with open(prediction_save_path, 'w') as f:
        json.dump(sample_generations, f)
    wandb.save(prediction_save_path)


def type_prediction(
    api_key: str,
    prediction_save_path: str = None,
    prediction_save_every_n_iters: int = 16,
    # gpt parameters
    engine: str = 'davinci-codex',
    gpt_n_shot: int = 0,
    gpt_temperature: float = 0.0,
    gpt_top_p: float = 1.0,
    gpt_frequency_penalty: float = 0.0,
    gpt_presence_penalty: float = 0.0,
    # data
    eval_filepath: str = TYPE_EVAL_DATA_FILEPATH,
    type_vocab_filepath: str = TYPE_VOCAB_FILEPATH,
    spm_filepath: str = SPM_UNIGRAM_FILEPATH,
    limit_dataset_size=-1,
    data_max_seq_len=-1,
    subword_regularization_alpha=0.0,
):
    config = locals()
    del config['api_key']
    wandb.init(name=f"openai-{engine}-{gpt_n_shot}shot", config=config, project="type_eval_openai", entity="ml4code")
    if prediction_save_path is None:
        prediction_save_path = wandb.run.dir + "/predictions.json"
    logger.info(f"Saving predictions to {prediction_save_path}")

    # Create codex API
    api = CodexAPI(api_key)

    # Load data
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")
    id_to_target, target_to_id = load_type_vocab(type_vocab_filepath)
    no_type_id = target_to_id["O"]
    any_id = target_to_id["$any$"]
    assert no_type_id == 0  # Just a sense check since O is the first line in the vocab file

    # Make dataloader
    logger.info(f"Eval data path {eval_filepath}")
    eval_dataset = DeepTyperDatasetText(
        eval_filepath,
        split_source_targets_by_tab=eval_filepath.endswith(".json"),
    )

    # Shuffle Dataset
    samples = [x for x in eval_dataset]
    random_order = np.random.permutation(len(samples))
    eval_dataset = [samples[i] for i in random_order[:limit_dataset_size]]

    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    # eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=0)

    # Metric
    def top1_acc(pred, gt, ignore_idx=(no_type_id,)):
        # pred and gt are a list of strings
        num_correct, num_total = 0, 0
        for p, g in zip(pred, gt):
            if g not in ignore_idx:
                num_total += 1
                if p == g:
                    num_correct += 1
        return num_correct, num_total

    # Evaluate
    sample_generations = []
    num1, num_labels_total = 0, 0
    num1_any, num_labels_any_total = 0, 0
    with tqdm(eval_dataset, desc="eval") as pbar:
        for X, labels in pbar:
            with Timer() as t:
                pred, gt = api.get_types(
                    X, labels, engine=engine, temperature=gpt_temperature, top_p=gpt_top_p,
                    frequency_penalty=gpt_frequency_penalty, presence_penalty=gpt_presence_penalty,
                    n_shot=gpt_n_shot)
                sample_generations.append(dict(code=X, labels=labels, pred_types=pred, gt_types=gt))

            # compute metrics
            corr1_any, num_any = top1_acc(pred, gt, ignore_idx=(no_type_id,))
            corr1_all, num_all = top1_acc(pred, gt, ignore_idx=(no_type_id, any_id))
            num1 += corr1_all
            num_labels_total += num_all
            num1_any += corr1_any
            num_labels_any_total += num_any

            pbar.set_description(
                f"eval avg acc1_any {num1_any / (num_labels_any_total + 1e-6) * 100:.2f}%"
                    + f", avg acc1 {num1 / (num_labels_total + 1e-6) * 100:.2f}%")
            wandb.log(dict(eval_time=t.interval, acc1_any=corr1_any / (num_any + 1e-6), acc1=corr1_all / (num_all + 1e-6)))

            # save examples
            if len(sample_generations) % prediction_save_every_n_iters == 0:
                with open(prediction_save_path, 'w') as f:
                    json.dump(sample_generations, f)
                wandb.save(prediction_save_path)

    # compute overall metrics
    # todo compute overall metrics
    # save final results
    with open(prediction_save_path, 'w') as f:
        json.dump(sample_generations, f)
    wandb.save(prediction_save_path)



if __name__ == "__main__":
    fire.Fire({'method_naming': method_naming, 'type_prediction': type_prediction})
