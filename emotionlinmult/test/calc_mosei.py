import os
import json
import numpy as np
import pandas as pd
from pprint import pprint
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv
from emotionlinmult.train.metrics import calculate_sentiment


def load_sentiment_predictions(pred_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions from a JSON file.

    Expected format:
    {
        "task1": {
            "sample_id1": {"y_pred": [...], "y_true": value, "mask": bool},
            "sample_id2": {...}
        },
        "task2": {...}
    }
    """
    with open(pred_file, 'r') as f:
        data = json.load(f)

    # sentiment
    preds_dict = {}
    targets_dict = {}
    masks_dict = {}
    for sample_id, sample_data in data['sentiment'].items():
        preds_dict[sample_id] = np.array(sample_data['y_pred'])
        targets_dict[sample_id] = np.array(sample_data['y_true'])
        masks_dict[sample_id] = np.array(sample_data['mask'])

    y_pred = np.array(list(preds_dict.values()))
    y_true = np.array(list(targets_dict.values()))
    mask = np.array(list(masks_dict.values()))

    assert np.all(mask)

    return y_pred, y_true, mask


if __name__ == '__main__':
    pred_sent_file = 'results/poster/SDSTL/stage2/poster_sdstl_mosei_40_cw2/_checkpoint/checkpoint_valid_combined_score.ckpt'
    y_pred_sent, y_true_sent, None = load_sentiment_predictions(pred_sent_file)
    metrics_sent = calculate_sentiment(y_pred_sent, y_true_sent, None) 
    print('Sentiment MAE:', metrics_sent['MAE'])
    print('Sentiment CORR:', metrics_sent['CORR'])
    print('Sentiment ACC_2:', metrics_sent['ACC_2'])
    print('Sentiment ACC_7:', metrics_sent['ACC_7'])
    print('Sentiment F1_2:', metrics_sent['F1_2'])
    print('Sentiment F1_7:', metrics_sent['F1_7'])

    pred_sent_file = 'results/poster/MDMTL/stage2/poster_mdmtl_40_cw/predictions/checkpoint_valid_combined_score/test_mosei.json'
    y_pred_sent, y_true_sent, None = load_sentiment_predictions(pred_sent_file)
    metrics_sent = calculate_sentiment(y_pred_sent, y_true_sent, None) 
    print('Sentiment MAE:', metrics_sent['MAE'])
    print('Sentiment CORR:', metrics_sent['CORR'])
    print('Sentiment ACC_2:', metrics_sent['ACC_2'])
    print('Sentiment ACC_7:', metrics_sent['ACC_7'])
    print('Sentiment F1_2:', metrics_sent['F1_2'])
    print('Sentiment F1_7:', metrics_sent['F1_7'])

    