#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/MTL/stage1/dataloader_all.yaml" \
    "--model_config_path" "configs/MTL/stage1/model.yaml" \
    "--train_config_path" "configs/MTL/stage1/train.yaml" \
    "--experiment_name" "all_nipg30_gpu0"
