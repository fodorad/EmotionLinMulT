#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDMTL/mead/dataloader_AV_ec_ei.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_40_AV.yaml" \
    "--train_config_path" "configs/SDMTL/train_ec_ei.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "mead_AV_ec_ei_40_all" \
    "--batch_size" "64" \
    "--devices" "1"
