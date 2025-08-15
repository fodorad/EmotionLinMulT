from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, mean_absolute_error, mean_squared_error
)
from typing import Optional, List, Dict, Union, Tuple


def format(x: float) -> float:
    """Format a float value to 3 decimal places."""
    return float(np.round(x, decimals=3))


def to_numpy(tensor_or_array) -> np.ndarray:
    """Convert a PyTorch tensor or NumPy array to NumPy array.
    
    Args:
        tensor_or_array: Input tensor or array
        
    Returns:
        np.ndarray: NumPy array
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return np.array(tensor_or_array)


def calculate_reconstruction(
        preds: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        output_path: str | Path = None
    ) -> dict:
    """Calculate reconstruction MSE"""
    preds = to_numpy(preds)
    targets = to_numpy(targets)

    return {'MSE': format(mean_squared_error(targets, preds))}


def calculate_sentiment(
        preds: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.BoolTensor | None = None,
    ) -> dict:
    """Calculate sentiment metrics for regression task.
    
    Args:
        preds: Predicted sentiment values, shape (N,)
        targets: Target sentiment values, shape (N,)
        mask: Boolean mask for valid frames (N,), optional
        
    Returns:
        dict: Dictionary containing the following metrics:
            - acc_7: 7-class accuracy (-3 to 3)
            - acc_2: Binary accuracy (positive/negative)
            - f1_7: 7-class weighted F1 score
            - f1_2: Binary weighted F1 score
            - mae: Mean absolute error
            - corr: Pearson correlation
    """
    preds = to_numpy(preds)
    targets = to_numpy(targets)

    if mask is not None:
        mask = to_numpy(mask)
        preds = preds[mask]
        targets = targets[mask]

    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = np.squeeze(preds, axis=-1) # (N, 1) -> (N,)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = np.squeeze(targets, axis=-1) # (N, 1) -> (N,)

    preds = np.clip(preds, a_min=-3, a_max=3)
    return sentiment_metrics(preds, targets)


def sentiment_metrics(
        preds: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
    """Calculate sentiment metrics for regression task.
    
    Args:
        preds: Predicted sentiment values, shape (N,)
        targets: Target sentiment values, shape (N,)
        
    Returns:
        dict: Dictionary containing the following metrics:
            - acc_7: 7-class accuracy (-3 to 3)
            - acc_2: Binary accuracy (positive/negative)
            - f1_7: 7-class weighted F1 score
            - f1_2: Binary weighted F1 score
            - mae: Mean absolute error
            - corr: Pearson correlation
    """
    assert preds.ndim == 1 and preds.shape == targets.shape, \
        f"Expected preds and targets to be 1D arrays of the same shape, got {preds.shape} and {targets.shape}"

    # 7-class metrics (-3 to 3)
    preds_7_class = np.round(preds).astype(int)
    targets_7_class = np.round(targets).astype(int)
    ACC_7 = format(accuracy_score(targets_7_class, preds_7_class))
    F1_7 = format(f1_score(targets_7_class, preds_7_class, average='weighted'))

    # Binary metrics (positive/negative) 
    preds_binary = (preds > 0).astype(int)
    targets_binary = (targets > 0).astype(int)
    ACC_2 = format(accuracy_score(targets_binary, preds_binary))
    F1_2 = format(f1_score(targets_binary, preds_binary, average='weighted'))

    # Regression metrics
    MAE = format(mean_absolute_error(targets, preds))
    CORR, _ = pearsonr(preds, targets)
    CORR = format(CORR)

    # Compile metrics
    metrics = {
        'ACC_7': ACC_7,
        'ACC_2': ACC_2,
        'F1_7': F1_7,
        'F1_2': F1_2,
        'MAE': MAE,
        'CORR': CORR
    }

    return metrics


def calculate_sentiment_class(
        logits: np.ndarray | torch.Tensor,  # Shape: (N, 3)
        targets: np.ndarray | torch.Tensor,  # Shape: (N,)
        output_path: str | Path = None
    ) -> dict:
    """Calculate metrics for sentiment classification task.
    
    Args:
        logits: Model logits/predictions, shape (N, 3) for 3 classes (neutral, positive, negative)
        targets: Target class indices, shape (N,)
        output_path: Optional path to save metrics as CSV
        
    Returns:
        dict: Classification metrics dictionary
    """
    # Convert all inputs to numpy
    logits = to_numpy(logits)    # Shape: (N, C)
    targets = to_numpy(targets)  # Shape: (N,)

    # Get predicted class indices
    preds = np.argmax(logits, axis=1)  # Shape: (N,)

    return classification_metrics(preds, targets, n_classes=3)


def calculate_emotion_intensity(
        logits: np.ndarray | torch.Tensor,  # Shape: (N, 3)
        targets: np.ndarray | torch.Tensor,  # Shape: (N,)
        output_path: str | Path = None
    ) -> dict:
    """Calculate metrics for sentiment classification task.
    
    Args:
        logits: Model logits/predictions, shape (N, 3) for 3 classes (neutral, positive, negative)
        targets: Target class indices, shape (N,)
        output_path: Optional path to save metrics as CSV
        
    Returns:
        dict: Classification metrics dictionary
    """
    # Convert all inputs to numpy
    logits = to_numpy(logits)    # Shape: (N, C)
    targets = to_numpy(targets)  # Shape: (N,)

    # Get predicted class indices
    preds = np.argmax(logits, axis=1)  # Shape: (N,)

    return classification_metrics(preds, targets, n_classes=3)


def calculate_emotion_class_from_logits(
        logits: np.ndarray | torch.Tensor,  # Shape: (N, 8) or (N, T, 8)
        targets: np.ndarray | torch.Tensor,  # Shape: (N,) or (N, T)
        masks: np.ndarray | torch.Tensor | None = None,  # Shape: (N,) or (N, T)
        is_framewise: bool = False,
    ) -> dict:
    """Calculate metrics for emotion classification task.

    Args:
        logits: Model logits, shape (N, C) or (N, T, C) for C emotion classes
        targets: Target class indices, shape (N,) or (N, T) with values in [0, C-1]
        masks: Boolean mask for valid frames (N,) or (N, T), optional
        is_framewise: Whether the targets are framewise

    Returns:
        dict: Classification metrics dictionary
    """
    # Convert all inputs to numpy
    logits = to_numpy(logits)    # Shape: (N, C) or (N, T, C)
    targets = to_numpy(targets)  # Shape: (N,) or (N, T)

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

    if masks is not None:
        assert masks.ndim == 1 or masks.ndim == 2, \
            f"Expected masks to be 1D or 2D array, got {masks.shape}"

        masks = to_numpy(masks)      # Shape: (N,) or (N, T)
        preds = preds[masks]
        targets = targets[masks]

    return classification_metrics(preds, targets, n_classes=8)


def classification_metrics(
        preds: np.ndarray,   # Shape: (N,)
        targets: np.ndarray,  # Shape: (N,)
        n_classes: int
    ) -> dict:
    """Calculate classification metrics.
    
    Args:
        preds: Model predictions as class indices, shape (N,) with values in [0, C-1]
        targets: Target class indices, shape (N,) with values in [0, C-1]
        n_classes: Number of classes C

    Returns:
        dict: Dictionary containing the following metrics:
            - ACC: Accuracy score
            - ConfusionMatrix: Flattened confusion matrix
            - NormalizedConfusionMatrix: Flattened normalized confusion matrix
            - Support: Class support counts
            - P: Weighted precision
            - R: Weighted recall
            - F1: Weighted F1 score
    """
    # Ensure correct shapes
    assert targets.ndim == 1 and targets.shape == preds.shape, \
        f"Expected targets and preds to be 1D arrays of the same shape, got {targets.shape} and {preds.shape}"

    # Convert to int type
    targets = targets.astype(int)
    preds = preds.astype(int)

    # Calculate metrics
    metrics = {}
    metrics["ACC"] = format(accuracy_score(targets, preds))

    # Confusion matrices
    cm = confusion_matrix(targets, preds, labels=np.arange(n_classes))
    metrics["ConfusionMatrix"] = list(cm.ravel())

    ncm = confusion_matrix(targets, preds, normalize="true", labels=np.arange(n_classes))
    metrics["NormalizedConfusionMatrix"] = list(ncm.ravel())

    # Class support
    metrics["Support"] = list(np.bincount(targets, minlength=n_classes))

    # Weighted metrics
    metrics["P"] = format(precision_score(targets, preds, average="weighted", zero_division=0))
    metrics["R"] = format(recall_score(targets, preds, average="weighted", zero_division=0))
    metrics["F1"] = format(f1_score(targets, preds, average="weighted", zero_division=0))
    metrics["F1_macro"] = format(f1_score(targets, preds, average="macro", zero_division=0))

    return metrics


def calculate_va(
    preds: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.BoolTensor | None = None,
    output_path: str | Path = None,
    aggregation: str = "all"
    ) -> dict:
    """Calculate valence/arousal metrics for video sequences.
    
    Args:
        preds: Predicted values (N, T, 1) or (N, T)
        targets: Target values (N, T, 1) or (N, T)
        mask: Boolean mask for valid frames (N, T), optional
        output_path: Optional path to save metrics as CSV
        aggregation: "all" (concat all frames) or "mean" (average per-video metrics)
        
    Returns:
        dict: VA metrics dictionary with CCC, RMSE, SAGR, PCC
    """
    # Convert to numpy and ensure proper shape
    preds = to_numpy(preds)
    targets = to_numpy(targets)
    if mask is not None:
        mask = to_numpy(mask)

    if aggregation == "all":
        # Global metrics across all frames
        if mask is not None:
            return va_metrics(preds[mask].reshape(-1), targets[mask].reshape(-1), output_path)
        else:
            return va_metrics(preds.reshape(-1), targets.reshape(-1), output_path)
    
    elif aggregation == "mean":
        # Calculate per-video metrics then average
        # Ensure shape (N, T)
        if preds.ndim == 3:
            preds = preds.squeeze(-1)  # (N, T, 1) => (N, T)
        if targets.ndim == 3:
            targets = targets.squeeze(-1)
        
        per_video_metrics = []
        if mask is not None:
            for video_preds, video_targets, video_mask in zip(preds, targets, mask):
                metrics = va_metrics(video_preds[video_mask].reshape(-1), video_targets[video_mask].reshape(-1))
                per_video_metrics.append(metrics)
        else:
            for video_preds, video_targets in zip(preds, targets):
                metrics = va_metrics(video_preds.reshape(-1), video_targets.reshape(-1))
                per_video_metrics.append(metrics)
            
        # Average across videos
        avg_metrics = {
            "CCC": format(np.mean([m["CCC"] for m in per_video_metrics])),
            "RMSE": format(np.mean([m["RMSE"] for m in per_video_metrics])),
            "SAGR": format(np.mean([m["SAGR"] for m in per_video_metrics])),
            "PCC": format(np.mean([m["PCC"] for m in per_video_metrics]))
        }
        
        if output_path:
            save_metrics(avg_metrics, per_video_metrics, output_path)
            
        return avg_metrics


def va_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    output_path: str | Path = None
    ) -> dict:
    """Core VA metric calculation for flattened frame predictions"""
    # Input validation
    assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} vs {targets.shape}"
    
    # Main metrics calculation (same as before)
    rmse = np.sqrt(np.mean((preds - targets)**2))
    sagr = np.mean(np.sign(preds) == np.sign(targets))
    pcc, _ = pearsonr(preds, targets)
    
    mean_pred = np.mean(preds)
    mean_gt = np.mean(targets)
    sample_cov = np.cov(preds, targets)[0,1]
    var_pred = np.var(preds)
    var_gt = np.var(targets)
    ccc = (2 * sample_cov) / (var_pred + var_gt + (mean_pred - mean_gt)**2 + 1e-8)
    
    population_cov = np.mean((targets - mean_gt) * (preds - mean_pred))
    icc31 = (2 * population_cov) / (var_gt + var_pred) # ICC(3,1)

    metrics = {
        "CCC": format(ccc),
        "RMSE": format(rmse),
        "SAGR": format(sagr),
        "PCC": format(pcc),
        "ICC": format(icc31)
    }

    return metrics


def save_metrics(metrics: dict, per_video: list, path: str | Path):
    """Save metrics to CSV with optional per-video results"""
    df = pd.DataFrame([metrics])
    if per_video:
        video_df = pd.DataFrame(per_video)
        video_df["video_id"] = video_df.index
        df = pd.concat([df, video_df], axis=1)
    df.to_csv(str(path), index=False)


def save_ncm_plot(
        ncm: np.ndarray, 
        class_names: List[str], 
        output_dir: str | Path, 
        dataset_name: str, 
        task_name: str
    ) -> None:
    """
    Save a normalized confusion matrix plot.
    
    Args:
        ncm: Normalized confusion matrix (n_classes, n_classes)
        class_names: List of class names
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset (for filename)
        task_name: Name of the task (for filename)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        ncm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
        linewidths=0.5
    )
    
    plt.title(f'Normalized Confusion Matrix\n{dataset_name} - {task_name}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'ncm_{dataset_name}_{task_name}.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved normalized confusion matrix to: {output_path}")


if __name__ == "__main__":
    # write a test for VA metrics
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(calculate_va_metrics(preds, targets, aggregation="all"))
    print(calculate_va_metrics(preds, targets, aggregation="mean")) 