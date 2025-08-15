#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDMTL/ravdess/dataloader_AV_ec_ei.yaml" \
    "--model_config_path" "configs/SDMTL/model_40_AV_uni.yaml" \
    "--train_config_path" "configs/SDMTL/train_ec_ei.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "ravdess_AV_ec_ei_uni_new" \
    "--batch_size" "64" \
    "--devices" "0"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDMTL/ravdess/dataloader_AV_ec_ei.yaml" \
    "--model_config_path" "configs/SDMTL/model_100_AV_uni.yaml" \
    "--train_config_path" "configs/SDMTL/train_ec_ei.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "ravdess_AV_ec_ei_uni_100" \
    "--batch_size" "16" \
    "--devices" "1"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDMTL/ravdess/dataloader_AV_ec_ei.yaml" \
    "--model_config_path" "configs/SDMTL/model_40_AV.yaml" \
    "--train_config_path" "configs/SDMTL/train_ec_ei.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "ravdess_AV_ec_ei" \
    "--batch_size" "64" \
    "--devices" "2"
