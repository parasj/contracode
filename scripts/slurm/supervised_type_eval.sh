#!/bin/bash
#SBATCH --job-name=type_eval
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_supervised_type_eval.log
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --time=4:00:00
#SBATCH --exclude=atlas,blaze,r16

# print host statistics
set -x
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
export BATCHSIZE=${BATCHSIZE:-1}
export SUBWORD_REGULARIZATION=${SUBWORD_REGULARIZATION:-0}

# set default arugments
[ -z "$CKPT" ] && { echo "Need to set CKPT"; exit 1; }
[ -z "$SUBWORD_REGULARIZATION" ] && { echo "Need to set SUBWORD_REGULARIZATION"; exit 1; }
[ -z "$BATCHSIZE" ] && { echo "Need to set BATCHSIZE"; exit 1; }
[ -f "$CKPT" ] || { echo "Checkpoint not found"; exit 1; }

# print argument names
echo "CKPT = $CKPT"
echo "BATCHSIZE = $BATCHSIZE"
echo ""
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER = $CUDA_DEVICE_ORDER"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "FREE_PORT = $FREE_PORT"

# load data to cache
mkdir -p $DATA_CACHE
chmod 755 $DATA_CACHE
rsync -avhW --no-compress --progress /work/paras/representjs/data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model $DATA_CACHE/codesearchnet_javascript/
rsync -avhW --no-compress --progress /work/paras/coderep/DeepTyper/data/target_wl $DATA_CACHE
rsync -avhW --no-compress --progress /work/paras/coderep/DeepTyper/data/valid_nounk.txt $DATA_CACHE
rsync -avhW --no-compress --progress /work/paras/coderep/DeepTyper/data/train_nounk.txt $DATA_CACHE
rsync -avhW --no-compress --progress /work/paras/coderep/DeepTyper/data/test_projects_gold_filtered.json $DATA_CACHE

# set up experiment dependencies
cd /work/paras/representjs
pip install torch
pip install -e .
npm install

python representjs/type_prediction.py eval \
  --eval_filepath "$DATA_CACHE/test_projects_gold_filtered.json" \
	--type_vocab_filepath $DATA_CACHE/target_wl \
	--spm_filepath $DATA_CACHE/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model \
	--resume_path "$CKPT" \
  --subword_regularization_alpha $SUBWORD_REGULARIZATION \
	--num_workers 4 \
	--batch_size $BATCHSIZE \
	--max_seq_len -1

