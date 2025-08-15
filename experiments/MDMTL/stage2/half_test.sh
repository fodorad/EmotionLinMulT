#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/MTL/stage2/dataloader.yaml" \
    "--model_config_path" "configs/MTL/stage2/model.yaml" \
    "--train_config_path" "configs/MTL/stage2/train.yaml" \
    "--experiment_name" "half_nipg31_gpu1" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/MTL/stage2/half_nipg31_gpu1/checkpoint/checkpoint_valid_loss.ckpt"
