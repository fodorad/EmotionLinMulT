#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/mosei/dataloader.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_40.yaml" \
    "--train_config_path" "configs/SDSTL/train_sentiment.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "mosei_sentiment_40" \
    "--batch_size" "64" \
    "--devices" "1"

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/SDSTL/mosei/dataloader.yaml" \
    "--model_config_path" "configs/MDMTL/stage2/model_100.yaml" \
    "--train_config_path" "configs/SDSTL/train_sentiment.yaml" \
    "--overwrite" "True" \
    "--experiment_name" "mosei_sentiment_100" \
    "--batch_size" "16" \
    "--devices" "2"
