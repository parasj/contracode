#!/bin/bash
#SBATCH --job-name=contrastive_pretrain
#SBATCH --output=/home/eecs/paras/slurm/coderep/%j_supervised_augment.log
#SBATCH --ntasks=1
#SBATCH --mem=256000
#SBATCH --time=125:00:00
#SBATCH --exclude=atlas,blaze,r16

date;hostname;pwd
free -mh

export PATH="/data/paras/miniconda3/bin:$PATH"
export DATA_CACHE="/dev/shm/paras"
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
rsync -avhW --no-compress --progress /work/paras/representjs/data/codesearchnet_javascript "$DATA_CACHE"

cd /work/paras/representjs
pip install torch
pip install -e .
npm install

python representjs/main.py train \
	--run_name 20017_augmentation_identifier_codeenc_noreset_4ldecoder \
	--program_mode augmentation --label_mode identifier \
	--n_decoder_layers=4 --subword_regularization_alpha 0.1 \
	--num_epochs 150 --save_every 10 --batch_size 32 --num_workers 16 --lr 1e-4 \
	--train_filepath $DATA_CACHE/codesearchnet_javascript/javascript_train_supervised.jsonl.gz \
	--eval_filepath $DATA_CACHE/codesearchnet_javascript/javascript_valid_0.jsonl.gz

