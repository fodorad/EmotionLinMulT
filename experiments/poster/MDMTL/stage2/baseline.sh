#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/baseline.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train_baseline_cw.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_mdmtl_baseline_cw" \
    "--batch_size" "32" \
    "--devices" "0"


python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/baseline.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train_baseline_cw.yaml" \
    "--experiment_name" "poster_mdmtl_baseline_cw" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/MDMTL/stage2/poster_mdmtl_baseline_cw/checkpoint/checkpoint_valid_combined_score.ckpt"