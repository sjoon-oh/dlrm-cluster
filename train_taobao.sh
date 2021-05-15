#!/bin/bash

# DLRM Facebookresearch Debloating
# author: sjoon-oh @ Github
# source: dlrm/our_train.py

dlrm_pt_bin="python3 dlrm_run.py"

dataset_base_dir="dataset"
dataset_dir="${dataset_base_dir}/user-behav-taobao"

dataset_proc_base_dir="dataset-processed"
dataset_proc_dir="${dataset_proc_base_dir}/user-behav-taobao"

if [ ! -d ${dataset_dir} ]; then
    mkdir ${dataset_base_dir}
    mkdir ${dataset_dir}
fi
if [ ! -d ${dataset_proc_dir} ]; then
    mkdir ${dataset_proc_base_dir}
    mkdir ${dataset_proc_dir}
fi

echo "run script (pytorch) ..."
$dlrm_pt_bin \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --raw-data-file="${dataset_dir}/train-taobao.txt" \
    --processed-data-file="${dataset_proc_dir}/taobao.npz" \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --print-freq=1024 \
    --test-freq=16384 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=2 \
    --save-model="model/model-kaggle.pt" \
    --den-feature-num=1 \
    --cat-feature-num=7

echo "done"