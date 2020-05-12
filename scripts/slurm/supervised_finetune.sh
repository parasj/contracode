#!/bin/bash
#SBATCH --job-name=contrastive_pretrain
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_supervised_finetune.log
#SBATCH --ntasks=1
#SBATCH --mem=256000
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16
set -x
date;hostname;pwd
free -mh

export PATH="/data/paras/miniconda3/bin:$PATH"
export DATA_CACHE="/data/paras/representjs_data"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER = $CUDA_DEVICE_ORDER"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
df -h
gpustat -cup
nvidia-smi
free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'
df -h | awk '$NF=="/"{printf "Disk Usage: %d/%dGB (%s)\n", $3,$2,$5}'
top -bn1 | grep load | awk '{printf "CPU Load: %.2f\n", $(NF-2)}'

chmod 755 -R ~/slurm

mkdir -p $DATA_CACHE
chmod 755 $DATA_CACHE
rsync -avhW --no-compress --progress /work/paras/representjs/data/codesearchnet_javascript "$DATA_CACHE"

cd /work/paras/representjs
pip install torch
pip install -e .
npm install

python representjs/main.py train --run_name 20020_identity_identifier_codeenc_noreset_finetune_10070s95k \
  --program_mode identity --label_mode identifier --n_decoder_layers=4 --subword_regularization_alpha 0 \
  --num_epochs 150 --save_every 5 --batch_size 16 --num_workers 8 --lr 1e-4 \
  --train_filepath $DATA_CACHE/codesearchnet_javascript/javascript_train_supervised.jsonl.gz \
  --eval_filepath $DATA_CACHE/codesearchnet_javascript/javascript_valid_0.jsonl.gz \
  --spm_filepath $DATA_CACHE/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model \
  --resume_path /work/paras/representjs/data/good_runs/10070/ckpt_pretrain_ep0002_step0095000.pth
