#!/bin/bash
#SBATCH --job-name=contrastive_pretrain_dist
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_pretrain_bert.log
#SBATCH --ntasks=1
#SBATCH --mem=400000
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16

set -x
date;hostname;pwd
free -mh

[ -z "$RUNNAME" ] && { echo "Need to set RUNNAME"; exit 1; }
[ -z "$BATCHSIZE" ] && { echo "Need to set BATCHSIZE"; exit 1; }
[ -z "$LR" ] && { echo "Need to set LR"; exit 1; }

export PATH="/data/paras/miniconda3/bin:$PATH"
export DATA_CACHE="/data/paras/data_cache"

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
rsync -avhW --no-compress --progress /work/paras/representjs/data/codesearchnet_javascript $DATA_CACHE
if [ ! -f "$DATA_CACHE/codesearchnet_javascript/javascript_augmented.pickle" ]; then
    (cd "$DATA_CACHE/codesearchnet_javascript" && gunzip -k "$DATA_CACHE/codesearchnet_javascript/javascript_augmented.pickle.gz")
fi

cd /work/paras/representjs
pip install torch
pip install -e .
npm install

python representjs/pretrain_distributed.py $RUNNAME --num_epochs=200 --batch_size=$BATCHSIZE --lr=$LR --num_workers=8 \
    --subword_regularization_alpha 0.1 --program_mode identity --loss_mode mlm --save_every 5000 \
    --train_filepath="$DATA_CACHE/codesearchnet_javascript/javascript_augmented.pickle" \
    --spm_filepath="$DATA_CACHE/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model" \
    --min_alternatives 1 --dist_url 'tcp://localhost:10005' --rank 0
