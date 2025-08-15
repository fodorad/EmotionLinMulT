#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_mdmtl_40" \
    "--batch_size" "32" \
    "--devices" "0"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train.yaml" \
    "--experiment_name" "poster_mdmtl_40" \
    "--batch_size" "32" \
    "--devices" "0" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/MDMTL/stage2/poster_mdmtl_40/poster_ckpt/checkpoint_valid_combined_score.ckpt"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train_cw.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_mdmtl_40_cw" \
    "--batch_size" "32" \
    "--devices" "0"



python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train_cw2.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_mdmtl_40_cw2" \
    "--batch_size" "32"