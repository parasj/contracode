#!/bin/bash
#SBATCH --job-name=evaluate_f1
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_evaluate_f1.log
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --time=3:00:00
#SBATCH --exclude=atlas,blaze,r16
#SBATCH --gres=gpu:1

set -x
date;hostname;pwd
free -mh

[ -z "$CHECKPOINT" ] && { echo "Need to set CHECKPOINT"; exit 1; }
export BATCHSIZE=${BATCHSIZE:-64}
export NUMDECODERLAYERS=${NUMDECODERLAYERS:-4}

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
rsync -avhW --no-compress --progress /work/paras/code/contracode/data/codesearchnet_javascript "$DATA_CACHE"

cd /work/paras/representjs
# pip install torch
# pip install -e .
# npm install

python representjs/main.py test -batch_size $BATCHSIZE --num_workers 8 \
  --n_decoder_layers $NUMDECODERLAYERS \
  --checkpoint_file $CHECKPOINT \
  --test_filepath $DATA_CACHE/codesearchnet_javascript/javascript_test_0.jsonl.gz \
  --spm_filepath $DATA_CACHE/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model \
  --beam_search_k 10 --per_node_k 4