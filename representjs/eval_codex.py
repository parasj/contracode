import ast
import json
import os
import time

import fire
from loguru import logger
import numpy as np
import openai
import pandas as pd
import sentencepiece as spm
import tqdm
import wandb

from data.jsonl_dataset import get_csnjs_dataset
from data.old_dataloader import javascript_dataloader
from data.util import Timer
from metrics.f1 import F1MetricMethodName


# Default argument values
DATA_DIR = "data/codesearchnet_javascript"
CSNJS_TEST_FILEPATH = os.path.join(DATA_DIR, "javascript_test_0.jsonl.gz")
SPM_UNIGRAM_FILEPATH = os.path.join(DATA_DIR, "csnjs_8k_9995p_unigram_url.model")


class CodexAPI:
    def __init__(self, api_key, min_interval=.01):
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
        return results['choices'][0]['text'].strip()


def method_naming(
    api_key: str,
    prediction_save_path: str,
    engine: str = 'davinci-codex',
    test_filepath: str = CSNJS_TEST_FILEPATH,
    spm_filepath: str = SPM_UNIGRAM_FILEPATH,
    limit_dataset_size=-1,
    program_mode="identity",
    label_mode="identifier",
):
    config = locals()
    del config['api_key']
    wandb.init(name=f"openai-{engine}", config=config, project="f1_eval", entity="ml4code")
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
    with tqdm.tqdm(test_loader, desc="test") as pbar:
        for X, Y, Z, ZZ in pbar:
            with Timer() as t:
                code = sp.Decode(X[0].numpy().tolist()).replace("[EOL]", "\n")
                pred = api.get_method_names(code, engine=engine)
                print("\n----------\n")
                # print("Code:\n", "  -->  \t" + code.replace('\n', '\n  -->  '), "\n")
                print("Pred: ", pred)
                print("GT:   ", sp.Decode(Y[0].numpy().tolist()))
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
            pbar.set_postfix(avg_metrics)
            wandb.log(avg_metrics, step=n_examples)
            wandb.log(item_metrics, step=n_examples)

    precision_avg = precision / n_examples
    recall_avg = recall / n_examples
    f1_overall = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
    logger.info(f"Precision: {precision_avg:.5f}%")
    logger.info(f"Recall: {recall_avg:.5f}%")
    logger.info(f"F1: {f1_overall:.5f}%")


    with open(prediction_save_path, 'w') as f:
        json.dump(sample_generations, f)
    wandb.save(prediction_save_path)


if __name__ == "__main__":
    fire.Fire({'method_naming': method_naming})