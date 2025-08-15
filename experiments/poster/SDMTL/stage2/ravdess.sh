#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDMTL/stage2/ravdess/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDMTL/stage2/ravdess/model_40.yaml" \
    "--train_config_path" "configs/poster/SDMTL/stage2/ravdess/train_cw2.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_sdmtl_ravdess_40_cw2" \
    "--batch_size" "32"