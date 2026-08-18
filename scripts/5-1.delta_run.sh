#!/bin/bash

dataset="mimic_cxr"
annotation="./data/LRRG/"
base_dir="./data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"

version="v1_delta"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 12 \
    --val_batch_size 16 \
    --freeze_vm True \
    --vis_use_lora True \
    --vis_r 16 \
    --vis_alpha 16 \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 16 \
    --devices 1 \
    --max_epochs 5 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log.txt