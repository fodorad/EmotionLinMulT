#!/bin/bash

python emotionlinmult/train/train.py \
    "--db_config_path" "configs/STL/sentiment/dataloader.yaml" \
    "--model_config_path" "configs/STL/sentiment/model.yaml" \
    "--train_config_path" "configs/STL/sentiment/train.yaml"