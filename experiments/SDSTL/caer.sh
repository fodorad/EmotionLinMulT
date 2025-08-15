#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/caer/dataloader.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/SDSTL/train_ec.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "caer_ec_40" \
    "--batch_size" "64" \
    "--devices" "0"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/caer/dataloader.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/SDSTL/train_ec_cw.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "caer_ec_40_cw" \
    "--batch_size" "64" \
    "--devices" "1"
