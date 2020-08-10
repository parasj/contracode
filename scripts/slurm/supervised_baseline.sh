#!/bin/bash
#SBATCH --job-name=contrastive_pretrain
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_supervised_baseline.log
#SBATCH --ntasks=1
#SBATCH --mem=256000
#SBATCH --time=125:00:00
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
export BATCHSIZE=${BATCHSIZE:-16}
export LR=${LR:-"1e-4"}
export PROGRAM_MODE=${PROGRAM_MODE:-"identity"}
export LABEL_MODE=${LABEL_MODE:-"identifier"}
export N_DECODER_LAYERS=${N_DECODER_LAYERS:-4}
export SUBWORD_REGULARIZATION=${SUBWORD_REGULARIZATION:-0}
export DATASET_LIMIT=${DATASET_LIMIT:-"-1"}

# set default arugments
[ -z "$RUNNAME" ] && { echo "Need to set RUNNAME"; exit 1; }
[ -z "$PROGRAM_MODE" ] && { echo "Need to set PROGRAM_MODE"; exit 1; }
[ -z "$LABEL_MODE" ] && { echo "Need to set LABEL_MODE"; exit 1; }
[ -z "$N_DECODER_LAYERS" ] && { echo "Need to set N_DECODER_LAYERS"; exit 1; }
[ -z "$SUBWORD_REGULARIZATION" ] && { echo "Need to set SUBWORD_REGULARIZATION"; exit 1; }
[ -z "$BATCHSIZE" ] && { echo "Need to set BATCHSIZE"; exit 1; }
[ -z "$LR" ] && { echo "Need to set LR"; exit 1; }
[ -z "$DATASET_LIMIT" ] && { echo "Need to set DATASET_LIMIT"; exit 1; }

# print argument names
echo "RUNNAME = $RUNNAME"
echo "PROGRAM_MODE = $PROGRAM_MODE"
echo "LABELMODE = $LABELMODE"
echo "LR = $LR"
echo "BATCHSIZE = $BATCHSIZE"
echo "DATASET_LIMIT = $DATASET_LIMIT"
echo ""
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER = $CUDA_DEVICE_ORDER"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "FREE_PORT = $FREE_PORT"

# load data to cache
mkdir -p $DATA_CACHE
chmod 755 $DATA_CACHE
rsync -avhW --no-compress --progress /work/paras/representjs/data/codesearchnet_javascript $DATA_CACHE

# set up experiment dependencies
cd /work/paras/representjs
pip install torch
pip install -e .
npm install

python representjs/main.py train \
	--run_name "$RUNNAME-$SLURM_JOB_ID" \
	--program_mode $PROGRAM_MODE --label_mode $LABEL_MODE \
	--n_decoder_layers=$N_DECODER_LAYERS --subword_regularization_alpha $SUBWORD_REGULARIZATION \
	--num_epochs 150 --save_every 5 --batch_size $BATCHSIZE --num_workers 8 --lr $LR \
	--train_filepath $DATA_CACHE/codesearchnet_javascript/javascript_train_supervised.jsonl.gz \
	--eval_filepath $DATA_CACHE/codesearchnet_javascript/javascript_valid_0.jsonl.gz
	--spm_filepath $DATA_CACHE/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model \
	--limit_dataset_size $DATASET_LIMIT
