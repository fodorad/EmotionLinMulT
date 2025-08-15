#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/MDMTL/stage2/dataloader_all.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/MDMTL/stage2/train.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "proposed_40" \
    "--batch_size" "64" \
    "--devices" "0"


