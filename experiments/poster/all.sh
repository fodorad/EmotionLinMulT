#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDMTL/stage2/afewva/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDMTL/stage2/afewva/model_40.yaml" \
    "--train_config_path" "configs/poster/SDMTL/stage2/afewva/train_cw2.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_sdmtl_afewva_40_cw2" \
    "--batch_size" "32"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDMTL/stage2/ravdess/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDMTL/stage2/ravdess/model_40.yaml" \
    "--train_config_path" "configs/poster/SDMTL/stage2/ravdess/train_cw2.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_sdmtl_ravdess_40_cw2" \
    "--batch_size" "32"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDMTL/stage2/cremad/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDMTL/stage2/cremad/model_40.yaml" \
    "--train_config_path" "configs/poster/SDMTL/stage2/cremad/train_cw2.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_sdmtl_cremad_40_cw2" \
    "--batch_size" "32"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDMTL/stage2/affwild2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDMTL/stage2/affwild2/model_40.yaml" \
    "--train_config_path" "configs/poster/SDMTL/stage2/affwild2/train_cw2.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_sdmtl_affwild2_40_cw2" \
    "--batch_size" "32"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDSTL/stage2/mosei/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDSTL/stage2/mosei/model_40.yaml" \
    "--train_config_path" "configs/poster/SDSTL/stage2/mosei/train_cw2.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "poster_sdstl_mosei_40_cw2" \
    "--batch_size" "32"
