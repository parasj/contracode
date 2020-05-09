#!/bin/bash
#SBATCH --job-name=contrastive_pretrain
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_pretrain.log
#SBATCH --ntasks=1
#SBATCH --mem=200gb
#SBATCH --gres=gpu:1
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16

date;hostname;pwd
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
export PATH="/data/paras/miniconda3/bin:$PATH"
chmod 755 -R ~/slurm

mkdir -p /tmp/data_cache
chmod 755 /tmp/data_cache
rsync -avhW --no-compress --progress /work/paras/representjs/data/codesearchnet_javascript /tmp/data_cache

cd /work/paras/representjs
pip install torch
pip install -e .
npm install
python representjs/pretrain.py 20012_full \
    --run_dir_base="/data/paras/coderep_runs" \
    --n_epochs=100 \
    --batch_size=96 \
    --lr="8e-4" \
    --num_workers=36 \
    --train_filepath="/tmp/data_cache/codesearchnet_javascript/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz" \
    --spm_filepath="/tmp/data_cache/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model"
#    --limit_dataset_size=10000 \