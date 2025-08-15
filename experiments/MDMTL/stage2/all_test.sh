#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/MTL/stage2/dataloader_all.yaml" \
    "--model_config_path" "configs/MTL/stage2/model.yaml" \
    "--train_config_path" "configs/MTL/stage2/train.yaml" \
    "--experiment_name" "all_nipg30_gpu1" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/MTL/stage2/all_nipg30_gpu1/checkpoint/checkpoint_valid_loss.ckpt"
