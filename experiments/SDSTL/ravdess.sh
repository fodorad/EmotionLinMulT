#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/ravdess/dataloader_ei.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/SDSTL/train_ei.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "ravdess_ei_40" \
    "--batch_size" "64" \
    "--devices" "0"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/ravdess/dataloader_ei.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_100.yaml" \
    "--train_config_path" "configs/SDSTL/train_ei.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "ravdess_ei_100" \
    "--batch_size" "16" \
    "--devices" "2"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/ravdess/dataloader_ec.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/SDSTL/train_ec.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "ravdess_ec_40" \
    "--batch_size" "64" \
    "--devices" "0"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/ravdess/dataloader_ec.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_100.yaml" \
    "--train_config_path" "configs/SDSTL/train_ec.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "ravdess_ec_100" \
    "--batch_size" "16" \
    "--devices" "2"
