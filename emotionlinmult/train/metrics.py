from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, mean_absolute_error
)


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


def calculate_sentiment(
        preds: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        output_path: str | Path = None
    ) -> dict:
    """Calculate sentiment metrics for regression task.
    
    Args:
        preds: Predicted sentiment values, shape (N,)
        targets: Target sentiment values, shape (N,)
        output_path: Optional path to save metrics as CSV
        
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

    preds = np.squeeze(preds, axis=-1) # (N, 1) -> (N,)
    preds = np.clip(preds, a_min=-3, a_max=3)
    return sentiment_metrics(preds, targets, output_path)


def sentiment_metrics(
        preds: np.ndarray,
        targets: np.ndarray,
        output_path: str | Path = None
    ) -> dict:
    """Calculate sentiment metrics for regression task.
    
    Args:
        preds: Predicted sentiment values, shape (N,)
        targets: Target sentiment values, shape (N,)
        output_path: Optional path to save metrics as CSV
        
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

    # Save to CSV if path provided
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({name: [value] for name, value in metrics.items()})
        df.to_csv(str(output_path), index=False)
        print(f'Metrics saved to {output_path}')
    
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


def calculate_emotion_class(
        logits: np.ndarray | torch.Tensor,  # Shape: (N, 7)
        targets: np.ndarray | torch.Tensor,  # Shape: (N,)
        output_path: str | Path = None
    ) -> dict:
    """Calculate metrics for emotion classification task.
    
    Args:
        logits: Model logits/predictions, shape (N, C) for C emotion classes
        targets: Target class indices, shape (N,) with values in [0, C-1]
        output_path: Optional path to save metrics as CSV
        
    Returns:
        dict: Classification metrics dictionary
    """
    # Convert all inputs to numpy
    logits = to_numpy(logits)    # Shape: (N, C)
    targets = to_numpy(targets)  # Shape: (N,)

    # Get predicted class indices
    preds = np.argmax(logits, axis=1)  # Shape: (N,)

    return classification_metrics(preds, targets, n_classes=7)

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

    return metrics



'''

def _classification_metrics(predictions, targets, mask, n_classes):
    """
    Calculates classification metrics (accuracy, precision, recall, and mean class-wise F1) with masking.

    Args:
        predictions (torch.Tensor): Predicted logits of shape (N, n_classes).
        targets (torch.Tensor): Ground truth labels of shape (N,) with values between 0 and n_classes - 1.
        mask (torch.Tensor): Binary mask tensor of shape (N,) where 1 indicates valid samples and 0 indicates invalid.
        n_classes (int): Number of classes.
    """

    # Initialize metrics
    accuracy = Accuracy(num_classes=n_classes, average="micro")
    precision = Precision(num_classes=n_classes, average="macro")
    recall = Recall(num_classes=n_classes, average="macro")
    f1_score = F1Score(num_classes=n_classes, average="macro")

    # Compute metrics
    acc = accuracy(pred_classes, valid_targets).item()
    p = precision(pred_classes, valid_targets).item()
    r = recall(pred_classes, valid_targets).item()
    f1 = f1_score(pred_classes, valid_targets).item()

    return {
        "ACC": acc,
        "P": p,
        "R": r,
        "F1": f1
    }


def va_metrics(predictions, targets, mask):
    """
    Calculates metrics (RMSE, SAGR, PCC, CCC) for valence or arousal predictions.

    Args:
        predictions (torch.Tensor): Predicted values of shape (B, T).
        targets (torch.Tensor): Ground truth values of shape (B, T).
        mask (torch.Tensor): Binary mask tensor of shape (B, T), where:
            - 1 indicates valid tokens
            - 0 indicates padded tokens.

    Returns:
        dict: A dictionary containing:
            - "rmse": Scalar, Root Mean Squared Error.
            - "sagr": Scalar, Sign Agreement Rate.
            - "pcc": Scalar, Pearson Correlation Coefficient.
            - "ccc": Scalar, Concordance Correlation Coefficient.
    """
    # Apply mask to flatten valid values
    valid_predictions = predictions[mask]
    valid_targets = targets[mask]

    # RMSE
    mse = torch.mean((valid_predictions - valid_targets) ** 2)
    rmse = torch.sqrt(mse).item()

    # SAGR
    sign_agreement = torch.sign(valid_predictions) == torch.sign(valid_targets)
    sagr = torch.mean(sign_agreement.float()).item()

    # PCC
    mean_pred = torch.mean(valid_predictions)
    mean_target = torch.mean(valid_targets)
    std_pred = torch.std(valid_predictions)
    std_target = torch.std(valid_targets)
    covariance = torch.mean((valid_predictions - mean_pred) * (valid_targets - mean_target))
    pcc = (covariance / (std_pred * std_target + 1e-8)).item()

    # CCC
    ccc = (2 * covariance / 
           (std_pred**2 + std_target**2 + (mean_pred - mean_target)**2 + 1e-8)).item()

    return {
        "rmse": rmse,
        "sagr": sagr,
        "pcc": pcc,
        "ccc": ccc
    }


def ACC_np(ground_truth, predictions):
    """Evaluates the mean accuracy
    """
    return np.mean(ground_truth.astype(int) == predictions.astype(int))


def RMSE(ground_truth, predictions, mask=None):
    """
    Evaluates the Root Mean Squared Error (RMSE) for PyTorch tensors.

    RMSE = sqrt(mean((GT - Pred)^2))

    Args:
        ground_truth (torch.Tensor): Ground truth tensor of shape (B, T) or (N,).
        predictions (torch.Tensor): Predictions tensor of the same shape as `ground_truth`.
        mask (torch.Tensor, optional): Binary mask tensor of shape (B, T), where:
            - 1 indicates valid tokens
            - 0 indicates padded tokens.
            If provided, only valid values are considered.

    Returns:
        torch.Tensor: A scalar tensor representing the RMSE.
    """
    if mask is not None:
        ground_truth = ground_truth[mask]
        predictions = predictions[mask]

    mse = torch.mean((ground_truth - predictions) ** 2)  # Mean squared error
    rmse = torch.sqrt(mse)  # Root mean squared error
    return rmse


def SAGR(ground_truth, predictions, mask=None):
    """
    Evaluates the Sign Agreement Rate (SAGR) for PyTorch tensors.

    SAGR = mean(sign(GT) == sign(Pred))

    Args:
        ground_truth (torch.Tensor): Ground truth tensor of shape (B, T) or (N,).
        predictions (torch.Tensor): Predictions tensor of the same shape as `ground_truth`.
        mask (torch.Tensor, optional): Binary mask tensor of shape (B, T), where:
            - 1 indicates valid tokens
            - 0 indicates padded tokens.
            If provided, only valid values are considered.

    Returns:
        torch.Tensor: A scalar tensor representing the SAGR.
    """
    if mask is not None:
        ground_truth = ground_truth[mask]
        predictions = predictions[mask]

    # Compute sign agreement
    sign_agreement = torch.sign(ground_truth) == torch.sign(predictions)
    sagr = torch.mean(sign_agreement.float())  # Convert boolean to float and calculate mean
    return sagr


def RMSE_np(ground_truth, predictions):
    """
        Evaluates the RMSE between estimate and ground truth.
    """
    return np.sqrt(np.mean((ground_truth-predictions)**2))


def SAGR_np(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    return np.mean(np.sign(ground_truth) == np.sign(predictions))


def PCC_np(ground_truth, predictions):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    return np.nan_to_num(np.corrcoef(ground_truth, predictions)[0,1])


def CCC_np(ground_truth, predictions):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    mean_pred = np.mean(predictions)
    mean_gt = np.mean(ground_truth)

    std_pred= np.std(predictions)
    std_gt = np.std(ground_truth)

    pearson = PCC(ground_truth, predictions)
    return 2.0*pearson*std_pred*std_gt/(std_pred**2+std_gt**2+(mean_pred-mean_gt)**2)


def PCC(ground_truth, predictions, mask=None):
    """
    Evaluates the Pearson Correlation Coefficient (PCC) for PyTorch tensors.
    
    PCC = Cov(GT, Pred) / (std(GT) * std(Pred))
    
    Args:
        ground_truth (torch.Tensor): Ground truth tensor of shape (B, T) or (N,), where:
            - B: Batch size
            - T: Time steps
            - N: Number of valid values after masking (if flattened).
        predictions (torch.Tensor): Predictions tensor of the same shape as `ground_truth`.
        mask (torch.Tensor, optional): Binary mask tensor of shape (B, T), with:
            - 1 indicating valid tokens
            - 0 indicating padded tokens.
            If provided, only valid values are considered.
    
    Returns:
        torch.Tensor: A scalar tensor representing the Pearson Correlation Coefficient (PCC).
        Value ranges from -1 to 1.
    """
    if mask is not None:
        # Apply mask to ground_truth and predictions
        ground_truth = ground_truth[mask]
        predictions = predictions[mask]

    # Compute mean, standard deviation, and covariance
    mean_gt = torch.mean(ground_truth)  # Scalar
    mean_pred = torch.mean(predictions)  # Scalar

    std_gt = torch.std(ground_truth)  # Scalar
    std_pred = torch.std(predictions)  # Scalar

    covariance = torch.mean((ground_truth - mean_gt) * (predictions - mean_pred))  # Scalar

    # Pearson Correlation Coefficient
    pcc = covariance / (std_gt * std_pred + 1e-8)  # Add epsilon to avoid division by zero
    return pcc


def CCC(ground_truth, predictions, mask=None):
    """
    Evaluates the Concordance Correlation Coefficient (CCC) for PyTorch tensors.
    
    CCC = 2 * PCC * std(GT) * std(Pred) / (std(GT)^2 + std(Pred)^2 + (mean(GT) - mean(Pred))^2)
    
    Args:
        ground_truth (torch.Tensor): Ground truth tensor of shape (B, T) or (N,), where:
            - B: Batch size
            - T: Time steps
            - N: Number of valid values after masking (if flattened).
        predictions (torch.Tensor): Predictions tensor of the same shape as `ground_truth`.
        mask (torch.Tensor, optional): Binary mask tensor of shape (B, T), with:
            - 1 indicating valid tokens
            - 0 indicating padded tokens.
            If provided, only valid values are considered.

    Returns:
        torch.Tensor: A scalar tensor representing the Concordance Correlation Coefficient (CCC).
        Value ranges from -1 to 1.
    """
    if mask is not None:
        # Apply mask to ground_truth and predictions
        ground_truth = ground_truth[mask]
        predictions = predictions[mask]

    # Compute mean, standard deviation, and PCC
    mean_gt = torch.mean(ground_truth)  # Scalar
    mean_pred = torch.mean(predictions)  # Scalar
    
    std_gt = torch.std(ground_truth)  # Scalar
    std_pred = torch.std(predictions)  # Scalar
    
    pearson = PCC(ground_truth, predictions)  # Use the PCC function; returns scalar
    
    # Concordance Correlation Coefficient
    ccc = (2 * pearson * std_gt * std_pred) / (
        std_gt**2 + std_pred**2 + (mean_gt - mean_pred)**2 + 1e-8
    )
    return ccc


def calculate_sentiment_metrics(preds_np, targets_np):
    # 7-class metrics
    # Round predictions to the nearest integer for classification into 7 classes (-3 to 3)
    preds_7_class = np.round(preds_np).astype(int)
    targets_7_class = np.round(targets_np).astype(int)

    acc_7 = accuracy_score(targets_7_class, preds_7_class)
    f1_7 = f1_score(targets_7_class, preds_7_class, average='weighted')

    # Binary metrics
    # Convert to binary: Sentiment > 0 is positive, <= 0 is negative
    preds_binary = (preds_np > 0).astype(int)
    targets_binary = (targets_np > 0).astype(int)

    acc_2 = accuracy_score(targets_binary, preds_binary)
    f1_2 = f1_score(targets_binary, preds_binary, average='weighted')

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(targets_np, preds_np)

    # Pearson Correlation (Corr)
    corr, _ = pearsonr(preds_np, targets_np)

    # Save results to a CSV file
    return {
        'acc_7': acc_7,
        'acc_2': acc_2,
        'f1_7': f1_7,
        'f1_2': f1_2,
        'mae': mae,
        'corr': corr
    }


def classification_metrics(y_true: torch.Tensor | np.ndarray,
                           y_score: torch.Tensor | np.ndarray,
                           n_classes: int):
    """Calculates classification metrics

    Note:
        y_true and y_score are the class indices.
        values are between 0 and C-1
        y_true expected shape is (N,)
        y_score expected shape is (N,)
        expected type is int
    """

    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().squeeze().numpy()

    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().squeeze().numpy()

    if y_true.ndim != 1:
        raise ValueError(f"Unexpected y_true shape. Expected is (N,) got instead {y_true.shape}")

    if y_score.ndim != 1:
        raise ValueError(f"Unexpected y_score shape. Expected is (N,) got instead {y_score.shape}")

    # Ensure correct type and shape
    y_true = y_true.astype(int)
    y_score = y_score.astype(int)

    # Metrics for multiclass classification
    metrics = {}
    metrics["Accuracy"] = format(accuracy_score(y_true, y_score))
    metrics["ConfusionMatrix"] = confusion_matrix(y_true, y_score, labels=np.arange(n_classes)).ravel()
    metrics["NormalizedConfusionMatrix"] = confusion_matrix(y_true, y_score, normalize="true", labels=np.arange(n_classes)).ravel()
    metrics["Support"] = np.bincount(y_true, minlength=n_classes)

    # Precision, Recall, F1 for each class
    # micro: Sum statistics over all labels
    # macro: Calculate statistics for each label and average them
    # weighted: calculates statistics for each label and computes weighted average using their support
    metrics["Precision"] = format(precision_score(y_true, y_score, average="micro", zero_division="warn"))
    metrics["Recall"] = format(recall_score(y_true, y_score, average="micro", zero_division="warn"))
    metrics["F1"] = format(f1_score(y_true, y_score, average="micro", zero_division="warn"))
    return metrics


if __name__ == "__main__":
    import pprint
    y_true = torch.Tensor([0, 1, 2, 0, 1, 2])
    y_prob = torch.Tensor([0, 2, 1, 0, 0, 1])
    metrics = classification_metrics(y_true, y_prob, 3)
    pprint.pprint(metrics)
    # expected f1 is 0.3333

'''