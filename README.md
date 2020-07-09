# Contrastive Code Representation Learning
By Paras Jain, Ajay Jain, Tianjun Zhang, Pieter Abbeel, Joseph E. Gonzalez and Ion Stoica

### **Learning functionality-based representations of programs**

<img src="https://contrastive-code.s3.amazonaws.com/img/teaser_figure.png">

Machine-aided programming tools such as type predictors and code summarizers are increasingly learning-based. However, most code representation learning approaches rely on supervised learning with task-specific annotated datasets. **We propose Contrastive Code Representation Learning (ContraCode), a self-supervised algorithm for learning task-agnostic semantic representations of programs via contrastive learning.**

Our approach uses no human-provided labels, relying only on the raw text of programs. In particular, we design an unsupervised pretext task by generating textually divergent copies of source functions via automated source-to-source compiler transforms that preserve semantics. We train a neural model to identify variants of an anchor program within a large batch of negatives. To solve this task, the network must extract program features representing the functionality, not form, of the program. This is the first application of instance discrimination to code representation learning to our knowledge. We pre-train ContraCode over 1.8m unannotated JavaScript methods mined from GitHub. ContraCode pre-training improves code summarization accuracy by 8% over supervised approaches and 5% over BERT pre-training. Moreover, our approach is agnostic to model architecture; for a type prediction task, contrastive pre-training consistently improves the accuracy of existing baselines.

This repository contains code to augment JavaScript programs with code transformations, pre-train LSTM and Transformer models with ContraCode, and to finetune the models on downstream tasks.

## Installation
Dependencies: Python 3.7, NodeJS, NPM
```bash
$ npm install
$ pip install -e "."
$ python scripts/download_data.py
```

## Instructions to train ContraCode MoCo baseline
### Pre-training MoCo model using ContraCode
```bash
$ representjs/pretrain_distributed.py 200226_pretrain_dist --num_epochs=200 \
    --batch_size=96 --lr=1e-4 --num_workers=6 --subword_regularization_alpha 0.1 \
    --program_mode contrastive --label_mode contrastive --save_every 5000 \
    --train_filepath=data/codesearchnet_javascript/javascript_augmented.pickle.gz \
    --spm_filepath=data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model \
    --min_alternatives 1 --dist_url tcp://localhost:10001 --rank 0
```

### Fine-tuning MoCo pretrained model on identifier prediction
```bash
$ representjs/main.py train --run_name 10120_identity_identifier_codeenc_noreset_finetune_4ldecoder_20026s45k \
    --program_mode identity --label_mode identifier --n_decoder_layers=4 --subword_regularization_alpha 0 \
    --num_epochs 100 --save_every 5 --batch_size 32 --num_workers 4 --lr 1e-4 \
    --train_filepath data/codesearchnet_javascript/javascript_train_supervised.jsonl.gz \
    --eval_filepath data/codesearchnet_javascript/javascript_valid_0.jsonl.gz \
    --resume_path PATH_TO_PRETRAIN_CKPT
```

## Citation
If you find this code or our paper relevant to your work, please cite our arXiv paper:
```
@article{jain2020contrastive,
  title={Contrastive Code Representation Learning},
  author={Paras Jain and Ajay Jain and Tianjun Zhang
  and Pieter Abbeel and Joseph E. Gonzalez and Ion Stoica},
  year={2020},
  journal={arXiv preprint}
}
```