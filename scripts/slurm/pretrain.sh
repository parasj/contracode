#!/bin/bash
#SBATCH --job-name=contrastive_pretrain
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_pretrain.log
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=atlas,blaze

date;hostname;pwd
export PATH="/data/paras/miniconda3/bin:$PATH"
chmod 755 -R ~/slurm


cd /data/paras/representjs
python representjs/pretrain.py 20005 --n_epochs=1 --batch_size=8 --lr="1e-2" --data_limit_size=1000 --num_gpus=1
