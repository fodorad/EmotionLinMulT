#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/model_40_ogd.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train_cw_ogd.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_mdmtl_40_cw_ogd" \
    "--batch_size" "32" \
    "--devices" "0"
