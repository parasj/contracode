# Contrastive Code Representation Learning
### **Learning robust structured representations of Javascript**

<img src="https://contrastive-code.s3.amazonaws.com/img/teaser_figure.png">

Machine-aided programming tools such as type predictors and code summarizersare increasingly learning-based. However, most code representation learning ap-proaches rely on supervised learning with task-specific annotated datasets.

Contrastive Code Representation Learning (ContraCode), a self-supervised algorithm for learning task-agnostic semantic representations of programs via contrastive learning. Our approach uses no human-provided labels, relying only on the raw text of programs.

## Installation
Dependencies: Python 3.7, NodeJS, NPM
```bash
$ bash download_data.sh
$ npm install
$ pip install -e "."
$ npm install terser
```

## Citation
```
@misc{jain2020contracode,
    title={Contrastive Code Representation Learning},
    author={Paras Jain and Ajay Jain and Tianjun Zhang and Pieter Abbeel and Joseph E. Gonzalez and Ion Stoica},
    year={2020}
}
```
