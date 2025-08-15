#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/affwild2/dataloader_va.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/SDSTL/train_va.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "affwild2_va_40" \
    "--batch_size" "64" \
    "--devices" "0"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/affwild2/dataloader_va.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_100.yaml" \
    "--train_config_path" "configs/SDSTL/train_va.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "affwild2_va_100" \
    "--batch_size" "16" \
    "--devices" "2"


python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/affwild2/dataloader_expr.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/SDSTL/train_ec_fw.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "affwild2_ec_fw_40" \
    "--batch_size" "64" \
    "--devices" "0"
