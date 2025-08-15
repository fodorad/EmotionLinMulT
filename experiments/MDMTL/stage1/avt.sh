#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/MTL/stage1/dataloader_avt.yaml" \
    "--model_config_path" "configs/MTL/stage1/model.yaml" \
    "--train_config_path" "configs/MTL/stage1/train.yaml" \
    "--experiment_name" "avt_nipg30_gpu0"
