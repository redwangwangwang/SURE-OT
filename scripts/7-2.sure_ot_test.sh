#!/bin/bash
set -euo pipefail

dataset="mimic_cxr"
annotation="./data/LRRG/"
base_dir="./data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
version="sure_ot_v1"
savepath="./save/${dataset}/${version}"
delta_file="${savepath}/checkpoints/REPLACE_WITH_CHECKPOINT.pth"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --test \
    --dataset "${dataset}" \
    --annotation "${annotation}" \
    --base_dir "${base_dir}" \
    --delta_file "${delta_file}" \
    --test_batch_size 16 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path "${savepath}" \
    --num_workers 12 \
    --devices 1 \
    --use_sure_ot True \
    --sure_ot_num_tokens 2 \
    --sure_ot_epsilon 0.07 \
    --sure_ot_tau 0.7 \
    --sure_ot_iters 40 \
    --sure_ot_spatial_weight 0.05 \
    --sure_ot_use_role_adapters True \
    --sure_ot_use_swap False \
    2>&1 | tee -a "${savepath}/test.log"
