# Unsupervised representation learning of Javascript methods

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