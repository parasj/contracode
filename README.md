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
$ bash download_data.sh
$ npm install
$ pip install -e "."
$ npm install terser
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
