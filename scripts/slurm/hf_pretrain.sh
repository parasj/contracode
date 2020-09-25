#!/bin/bash
#SBATCH --job-name=hf_pretrain_bert
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_hf_pretrain_bert.log
#SBATCH --ntasks=1
#SBATCH --mem=400000
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16

set -x

# print host statistics
date;hostname;pwd
free -mh
df -h
gpustat -cup
nvidia-smi
free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'
df -h | awk '$NF=="/"{printf "Disk Usage: %d/%dGB (%s)\n", $3,$2,$5}'
top -bn1 | grep load | awk '{printf "CPU Load: %.2f\n", $(NF-2)}' 
chmod 755 -R ~/slurm

# program-wide constants
export PATH="/data/paras/miniconda3/bin:$PATH"
export DATA_CACHE="/data/paras/data_cache"
export FREE_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')";

# argument parsing
export BATCHSIZE=${BATCHSIZE:-64}
export LR=${LR:-"4e-4"}
[ -z "$RUNNAME" ] && { echo "Need to set RUNNAME"; exit 1; }
[ -z "$BATCHSIZE" ] && { echo "Need to set BATCHSIZE"; exit 1; }
[ -z "$LR" ] && { echo "Need to set LR"; exit 1; }

echo "LR = $LR"
echo "BATCHSIZE = $BATCHSIZE"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER = $CUDA_DEVICE_ORDER"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "FREE_PORT = $FREE_PORT"

# set up experiment dependencies
cd /work/paras/contracode
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
python representjs/hugging_face/run_language_modeling.py \
	--output_dir ./runs/$RUNNAME \
	--model_type roberta \
	--mlm \
	--tokenizer_name="./data/vocab/8k_bpe/8k_bpe-vocab.txt" \
	--do_train \
    --do_eval \
	--learning_rate $LR \
	--per_gpu_train_batch_size $BATCHSIZE \
	--num_train_epochs 5 \
	--save_total_limit 2 \
	--save_steps 2000 \
	--evaluate_during_training \
	--seed 42 \
	--train_data_file "$DATA_CACHE/hf_data/augmented_pretrain_df.train.txt" \
	--eval_data_file "$DATA_CACHE/hf_data/augmented_pretrain_df.test.txt"