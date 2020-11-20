#!/bin/bash
#SBATCH --job-name=pretrain_dist_bert_nonaug
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_pretrain_bert_nonaug.log
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

# load data to cache
mkdir -p $DATA_CACHE
chmod 755 $DATA_CACHE
rsync -avhW --no-compress --progress /work/paras/code/contracode/data/codesearchnet_javascript $DATA_CACHE
if [ ! -f "$DATA_CACHE/codesearchnet_javascript/javascript_augmented.pickle" ]; then
    (cd "$DATA_CACHE/codesearchnet_javascript" && gunzip -k "$DATA_CACHE/codesearchnet_javascript/javascript_augmented.pickle.gz")
fi
if [ ! -f "$DATA_CACHE/codesearchnet_javascript/javascript_nonaugmented_train.pickle" ]; then
  (cd "$DATA_CACHE/codesearchnet_javascript" && gunzip -k "$DATA_CACHE/codesearchnet_javascript/javascript_nonaugmented_train.pickle.gz")
fi

# set up experiment dependencies
cd /work/paras/code/contracode/
pip install torch
pip install -e .
npm install

# run train script
python representjs/pretrain_distributed.py $RUNNAME --num_epochs=200 --batch_size=$BATCHSIZE --lr=$LR --num_workers=8 \
    --subword_regularization_alpha 0. --program_mode identity --loss_mode mlm --save_every 5000 \
    --train_filepath="$DATA_CACHE/codesearchnet_javascript/javascript_nonaugmented_train.pickle" \
    --spm_filepath="$DATA_CACHE/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model" \
    --min_alternatives 1 --dist_url "tcp://localhost:$FREE_PORT" --rank 0 \
    --n_encoder_layers $N_ENCODER_LAYERS --d_model $D_MODEL --n_head $N_HEAD

#    --train_filepath="$DATA_CACHE/codesearchnet_javascript/javascript_augmented.pickle" \

