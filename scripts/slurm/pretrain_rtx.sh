#!/bin/bash
#SBATCH --job-name=contrastive_pretrain
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_pretrain.log
#SBATCH --ntasks=1
#SBATCH --mem=256000
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16,steropes

date;hostname;pwd
free -mh

export PATH="/data/paras/miniconda3/bin:$PATH"

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

mkdir -p /tmp/paras/data_cache
chmod 755 /tmp/paras/data_cache
rsync -avhW --no-compress --progress /work/paras/representjs/data/codesearchnet_javascript /tmp/paras/data_cache

cd /work/paras/representjs
pip install torch
pip install -e .
npm install
python representjs/pretrain.py 20013_full_steropes \
    --run_dir_base="/data/paras/coderep_runs" \
    --n_epochs=100 \
    --batch_size=64 \
    --lr="1e-4" \
    --num_workers=64 \
    --train_filepath="/tmp/paras/data_cache/codesearchnet_javascript/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz" \
    --spm_filepath="/tmp/paras/data_cache/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model" \
    --limit_dataset_size=400000
