#!/bin/bash
#SBATCH --job-name=contrastive_pretrain
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_pretrain.log
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=atlas,blaze,r16

date;hostname;pwd
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
export PATH="/data/paras/miniconda3/bin:$PATH"
chmod 755 -R ~/slurm

mkdir -p /tmp/data_cache
rsync -avhW --no-compress --progress /work/paras/representjs/data/codesearchnet_javascript /tmp/data_cache

cd /work/paras/representjs
pip install torch
pip install -e .
npm install
python representjs/pretrain.py 20008_pretrain_ace \
    --n_epochs=10 --batch_size=64 --lr="1e-3" \
    --data_limit_size=100000 --run_dir_base="/data/paras/coderep_runs" \
    --num_workers=4 --dataset="/tmp/data_cache/codesearchnet_javascript/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz" \
    --vocab="/tmp/data_cache/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model"