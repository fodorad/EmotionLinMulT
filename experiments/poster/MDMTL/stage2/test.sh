#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_a.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train_cw.yaml" \
    "--experiment_name" "poster_mdmtl_40_cw" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/MDMTL/stage2/poster_mdmtl_40_cw/_checkpoint/checkpoint_valid_combined_score.ckpt"

