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
from emotionlinmult.train.metrics import calculate_va


from emotionlinmult.preprocess import RAVDESS_EMOTION_MAPPING


def load_va_predictions(pred_file: str, task: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # VA
    preds_dict = {}
    targets_dict = {}
    masks_dict = {}
    for sample_id, sample_data in data[task].items():
        preds_dict[sample_id] = np.array(list(sample_data['y_pred'].values()))
        targets_dict[sample_id] = np.array(list(sample_data['y_true'].values()))
        masks_dict[sample_id] = np.array(list(sample_data['mask'].values()))

    y_pred = np.array(list(preds_dict.values()))
    y_true = np.array(list(targets_dict.values()))
    mask = np.array(list(masks_dict.values()))

    return y_pred, y_true, mask


if __name__ == '__main__':
    pred_file = 'results/poster/MDMTL/stage2/poster_mdmtl_40_cw/predictions/checkpoint_valid_combined_score/test_afew-va_face.json'
    y_pred_v, y_true_v, mask_v = load_va_predictions(pred_file, 'valence')
    y_pred_a, y_true_a, mask_a = load_va_predictions(pred_file, 'arousal')
    metrics_v = calculate_va(y_pred_v, y_true_v, mask_v)
    metrics_a = calculate_va(y_pred_a, y_true_a, mask_a)
    print('V PCC:', metrics_v['PCC'])
    print('A PCC:', metrics_a['PCC'])
    print('V CCC:', metrics_v['CCC'])
    print('A CCC:', metrics_a['CCC'])

    