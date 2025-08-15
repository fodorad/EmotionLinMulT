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
from emotionlinmult.train.metrics import classification_metrics


from emotionlinmult.preprocess import RAVDESS_EMOTION_MAPPING


def load_ec_predictions(pred_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # EC
    preds_dict = {}
    targets_dict = {}
    masks_dict = {}
    for sample_id, sample_data in data['emotion_class'].items():
        preds_dict[sample_id] = np.array(sample_data['y_pred'])
        targets_dict[sample_id] = np.array(sample_data['y_true'])
        masks_dict[sample_id] = np.array(sample_data['mask'])

    y_pred = np.array(list(preds_dict.values()))
    y_true = np.array(list(targets_dict.values()))
    mask = np.array(list(masks_dict.values()))

    assert np.all(mask)

    return y_pred, y_true, mask


def load_ei_predictions(pred_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # EI
    preds_dict = {}
    targets_dict = {}
    masks_dict = {}
    for sample_id, sample_data in data['emotion_intensity'].items():
        preds_dict[sample_id] = np.array(sample_data['y_pred'])
        targets_dict[sample_id] = np.array(sample_data['y_true'])
        masks_dict[sample_id] = np.array(sample_data['mask'])

    y_pred = np.array(list(preds_dict.values()))
    y_true = np.array(list(targets_dict.values()))
    mask = np.array(list(masks_dict.values()))

    assert np.all(mask)

    return y_pred, y_true, mask


def calculate_emotion_class(
        logits: np.ndarray | torch.Tensor,  # Shape: (N, 8) or (N, T, 8)
        targets: np.ndarray | torch.Tensor,  # Shape: (N,) or (N, T)
        mask: np.ndarray | torch.Tensor | None = None,  # Shape: (N,) or (N, T)
        is_framewise: bool = False,
        n_classes: int = 7, # poster
    ) -> dict:
    """Calculate metrics for emotion classification task.

    Args:
        logits: Model predictions, shape (N, C) or (N, T, C) for C emotion classes
        targets: Target class indices, shape (N,) or (N, T) with values in [0, C-1]
        masks: Boolean mask for valid frames (N,) or (N, T), optional
        is_framewise: Whether the targets are framewise

    Returns:
        dict: Classification metrics dictionary
    """

    if is_framewise:
        assert logits.ndim == 3 and targets.ndim == 2, \
            f"Expected logits and targets to be 3D and 2D arrays, got {logits.shape} and {targets.shape}"
        # Get predicted class indices
        preds = np.argmax(logits, axis=2)  # Shape: (N, T)
    else:
        assert logits.ndim == 2 and targets.ndim == 1, \
            f"Expected logits and targets to be 2D and 1D arrays, got {logits.shape} and {targets.shape}"
        # Get predicted class indices
        preds = np.argmax(logits, axis=1)  # Shape: (N,)

    if mask is not None:
        assert mask.ndim == 1 or mask.ndim == 2, \
            f"Expected mask to be 1D or 2D array, got {mask.shape}"

        preds = preds[mask]
        targets = targets[mask]

    return classification_metrics(preds, targets, n_classes=n_classes)


if __name__ == '__main__':
    pred_ec_file = 'results/poster/MDMTL/stage2/poster_mdmtl_40_cw/predictions/checkpoint_valid_combined_score/test_crema-d_expr.json'
    y_pred_ec, y_true_ec, _ = load_ec_predictions(pred_ec_file)
    metrics_ec = calculate_emotion_class(y_pred_ec, y_true_ec, None, n_classes=6)
    pred_ei_file = 'results/poster/MDMTL/stage2/poster_mdmtl_40_cw/predictions/checkpoint_valid_combined_score/test_crema-d_int.json'
    y_pred_ei, y_true_ei, _ = load_ei_predictions(pred_ei_file)
    metrics_ei = calculate_emotion_class(y_pred_ei, y_true_ei, None, n_classes=3)
    print('EC F1:', metrics_ec['F1'])
    print('EI F1:', metrics_ei['F1'])

    