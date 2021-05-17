#!/bin/bash

# DLRM Facebookresearch Debloating
# author: sjoon-oh @ Github
# source: dlrm/our_test_gpu.py

dlrm_pt_bin="python3 dlrm_run.py"

cluster_size=16

dataset_base_dir="dataset"
dataset_dir="${dataset_base_dir}/Kaggle"

dataset_proc_base_dir="dataset-processed"
dataset_proc_dir="${dataset_proc_base_dir}/Kaggle"

model_base_dir="model"
model_dir="${model_base_dir}/model-kaggle-${cluster_size}.pt"

if [ ! -d ${dataset_dir} ]; then
    mkdir ${dataset_base_dir}
    mkdir ${dataset_dir}
fi
if [ ! -d ${dataset_proc_dir} ]; then
    mkdir ${dataset_proc_base_dir}
    mkdir ${dataset_proc_dir}
fi

echo "run inference script (pytorch) ..."
$dlrm_pt_bin \
    --arch-sparse-feature-size=64 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --raw-data-file="${dataset_dir}/train.txt" \
    --processed-data-file="${dataset_proc_dir}/kaggle.npz" \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --print-freq=1024 \
    --test-freq=16384 \
    --print-time \
    --test-mini-batch-size=1024 \
    --test-num-workers=4 \
    --load-model="${model_dir}" \
    --inference-only \
    --use-gpu \
    --den-feature-num=13 \
    --cat-feature-num=26 \
    --clusters=${cluster_size} \
    --ignore-transfer-map="no" \
    --idx-2-gpu 2 11 15 20

echo "done"