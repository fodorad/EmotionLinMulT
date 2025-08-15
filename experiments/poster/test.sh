#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDMTL/stage2/afewva/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDMTL/stage2/afewva/model_40.yaml" \
    "--train_config_path" "configs/poster/SDMTL/stage2/afewva/train_cw2.yaml" \
    "--experiment_name" "poster_sdmtl_afewva_40_cw2" \
    "--batch_size" "32" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/SDMTL/stage2/poster_sdmtl_afewva_40_cw2/checkpoint/checkpoint_valid_combined_score.ckpt"


python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDMTL/stage2/ravdess/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDMTL/stage2/ravdess/model_40.yaml" \
    "--train_config_path" "configs/poster/SDMTL/stage2/ravdess/train_cw2.yaml" \
    "--experiment_name" "poster_sdmtl_ravdess_40_cw2" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/SDMTL/stage2/poster_sdmtl_ravdess_40_cw2/checkpoint/checkpoint_valid_combined_score.ckpt"


python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDMTL/stage2/cremad/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDMTL/stage2/cremad/model_40.yaml" \
    "--train_config_path" "configs/poster/SDMTL/stage2/cremad/train_cw2.yaml" \
    "--experiment_name" "poster_sdmtl_cremad_40_cw2" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/SDMTL/stage2/poster_sdmtl_cremad_40_cw2/checkpoint/checkpoint_valid_combined_score.ckpt"


python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDMTL/stage2/affwild2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDMTL/stage2/affwild2/model_40.yaml" \
    "--train_config_path" "configs/poster/SDMTL/stage2/affwild2/train_cw2.yaml" \
    "--experiment_name" "poster_sdmtl_affwild2_40_cw2" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/SDMTL/stage2/poster_sdmtl_affwild2_40_cw2/_checkpoint/checkpoint_valid_combined_score.ckpt"


python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/SDSTL/stage2/mosei/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/SDSTL/stage2/mosei/model_40.yaml" \
    "--train_config_path" "configs/poster/SDSTL/stage2/mosei/train_cw2.yaml" \
    "--experiment_name" "poster_sdstl_mosei_40_cw2" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/SDSTL/stage2/poster_sdstl_mosei_40_cw2/checkpoint/checkpoint_valid_combined_score.ckpt"


# MDMTL
python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train.yaml" \
    "--experiment_name" "poster_mdmtl_40" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/MDMTL/stage2/poster_mdmtl_40/checkpoint/checkpoint_valid_combined_score.ckpt"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train_cw.yaml" \
    "--experiment_name" "poster_mdmtl_40_cw" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/MDMTL/stage2/poster_mdmtl_40_cw/checkpoint/checkpoint_valid_combined_score.ckpt"



python emotionlinmult/train/train.py \
    "--db_config_path" "configs/poster/MDMTL/stage2/dataloader_av.yaml" \
    "--model_config_path" "configs/poster/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/poster/MDMTL/stage2/train_cw.yaml" \
    "--experiment_name" "poster_mdmtl_40_cw" \
    "--test_only" "True" \
    "--model_stage2_cp_path" "results/poster/MDMTL/stage2/poster_mdmtl_40_cw/checkpoint/checkpoint_valid_combined_score.ckpt"