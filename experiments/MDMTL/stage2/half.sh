#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/MTL/stage2/dataloader_half.yaml" \
    "--model_config_path" "configs/MTL/stage2/model.yaml" \
    "--train_config_path" "configs/MTL/stage2/train.yaml" \
    "--experiment_name" "half_nipg31_gpu1"
