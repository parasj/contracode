#!/bin/bash
set -x

# print host statistics
date;hostname;pwd
free -mh
df -h
nvidia-smi
free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'
df -h | awk '$NF=="/"{printf "Disk Usage: %d/%dGB (%s)\n", $3,$2,$5}'
top -bn1 | grep load | awk '{printf "CPU Load: %.2f\n", $(NF-2)}' 
chmod 755 -R ~/slurm

# program-wide constants
export PATH="/home/ajay/miniconda3/envs/paras/bin:$PATH"
export DATA_CACHE="/dev/shm/.contracode_cache"
export FREE_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')";

# argument parsing
export BATCHSIZEPERGPU=${BATCHSIZEPERGPU:-4}
export LR=${LR:-"1e-4"}
[ -z "$RUNNAME" ] && { echo "Need to set RUNNAME"; exit 1; }
[ -z "$BATCHSIZEPERGPU" ] && { echo "Need to set BATCHSIZEPERGPU"; exit 1; }
[ -z "$LR" ] && { echo "Need to set LR"; exit 1; }

echo "LR = $LR"
echo "BATCHSIZEPERGPU = $BATCHSIZEPERGPU"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER = $CUDA_DEVICE_ORDER"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "FREE_PORT = $FREE_PORT"

# set up experiment dependencies
cd ~/contracode
pip install torch
pip install -e .
npm install

# load data to cache
mkdir -p $DATA_CACHE
chmod 755 $DATA_CACHE
python scripts/download_data.py $DATA_CACHE --skip-csn

mkdir -p $DATA_CACHE/hf_cache
export TRANSFORMERS_CACHE=$DATA_CACHE/hf_cache
pwd

# run train script
python3 representjs/hugging_face/run_language_modeling.py \
	--output_dir ./runs/$RUNNAME \
	--model_type roberta \
	--mlm \
	--tokenizer_name="./data/vocab/8k_bpe/8k_bpe-vocab.txt" \
	--do_train \
    --do_eval \
	--learning_rate $LR \
	--per_gpu_train_batch_size $BATCHSIZEPERGPU \
	--num_train_epochs 5 \
	--save_total_limit 2 \
	--save_steps 2000 \
	--evaluate_during_training \
	--seed 42 \
	--train_data_file "$DATA_CACHE/hf_data/augmented_pretrain_df.train.txt" \
	--eval_data_file "$DATA_CACHE/hf_data/augmented_pretrain_df.test.txt"