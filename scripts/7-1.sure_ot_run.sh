#!/bin/bash
set -euo pipefail

dataset="mimic_cxr"
annotation="./data/LRRG/"
base_dir="./data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
version="sure_ot_v1"
savepath="./save/${dataset}/${version}"

mkdir -p "${savepath}"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --dataset "${dataset}" \
    --annotation "${annotation}" \
    --base_dir "${base_dir}" \
    --batch_size 8 \
    --val_batch_size 8 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path "${savepath}" \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    --max_epochs 5 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    --use_sure_ot True \
    --sure_ot_num_tokens 2 \
    --sure_ot_epsilon 0.07 \
    --sure_ot_tau 0.7 \
    --sure_ot_iters 40 \
    --sure_ot_spatial_weight 0.05 \
    --sure_ot_use_role_adapters True \
    --sure_ot_use_swap True \
    --lambda_sure_ot_swap 0.1 \
    --lambda_sure_ot_transport 0.01 \
    --lambda_sure_ot_reg 0.01 \
    2>&1 | tee -a "${savepath}/log.txt"
