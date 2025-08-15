import torch
import torch.nn.functional as F
from typing import Any
import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    source: https://github.com/Mikoto10032/AutomaticWeightedLoss

    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x: list[torch.Tensor]):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def compute_sentiment_class_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.BoolTensor,
        loss_fn: str,
        num_classes: int = 3,
        weight: float = 1.0,
        class_weights: torch.Tensor | list[float] | None = None
    ) -> torch.Tensor:
    """Compute sentiment classification loss."""
    if not mask.any():
        return torch.zeros(size=(), device=y_pred.device)

    y_pred = y_pred[mask]
    y_true = y_true[mask]

    if isinstance(class_weights, list):
        class_weights = torch.tensor(class_weights, device=y_pred.device)

    if loss_fn == "cross_entropy":
        return weight * F.cross_entropy(y_pred, y_true, weight=class_weights)
    elif loss_fn == "focal":
        return weight * FocalLoss(gamma=2, alpha=class_weights, task_type='multi-class', num_classes=num_classes)(y_pred, y_true)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def compute_sentiment_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.BoolTensor,
        weight: float = 1.0
    ) -> torch.Tensor:
    """Compute sentiment regression loss."""
    if not mask.any():
        return torch.zeros(size=(), device=y_pred.device)

    y_pred = y_pred[mask]
    y_true = y_true[mask]
    #return weight * torch.nn.functional.l1_loss(y_pred, y_true)
    #return weight * bell_l2_l1_loss(y_pred, y_true)

    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.view(-1)
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.view(-1)
    
    return weight * torch.nn.functional.mse_loss(y_pred, y_true)


def compute_sentiment_consistency_loss(
        pred_cls: torch.Tensor,
        pred_reg: torch.Tensor,
        weight: float = 1.0
    ) -> torch.Tensor:
    """Compute consistency loss between sentiment classification and regression."""
    pred_cls_probs = F.softmax(pred_cls, dim=1) # (B, 3)
    
    # class 0 : neutral
    # class 1 : positive
    # class 2 : negative
    # no constraints for neutral

    # For positive sentiment (class 1), regression should be > 0
    positive_mask = pred_cls_probs[:, 1] > 0.5
    positive_loss = F.relu(-pred_reg[positive_mask]).mean() if positive_mask.any() else torch.tensor(0.0, device=pred_cls.device)

    # For negative sentiment (class 2), regression should be < 0
    negative_mask = pred_cls_probs[:, 2] > 0.5
    negative_loss = F.relu(pred_reg[negative_mask]).mean() if negative_mask.any() else torch.tensor(0.0, device=pred_cls.device)

    # Total consistency loss
    consistency_loss = positive_loss + negative_loss

    return weight * consistency_loss


def compute_emotion_class_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.BoolTensor,
        loss_fn: str,
        num_classes: int = 8,
        weight: float = 1.0,
        is_framewise: bool = False,
        class_weights: torch.Tensor | list[float] | None = None
    ) -> torch.Tensor:
    """Compute emotion classification loss."""
    if not mask.any():
        return torch.zeros(size=(), device=y_pred.device)

    if is_framewise:
        # Reshape to (B*T, num_classes) for frame-wise prediction
        y_pred = y_pred[mask].view(-1, num_classes)
        y_true = y_true[mask].view(-1)
    else:
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    if isinstance(class_weights, list):
        class_weights = torch.tensor(class_weights, device=y_pred.device)

    if loss_fn == "cross_entropy":
        return weight * F.cross_entropy(y_pred, y_true, weight=class_weights)
    elif loss_fn == "focal":
        return weight * FocalLoss(gamma=2, alpha=class_weights, task_type='multi-class', num_classes=num_classes)(y_pred, y_true)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def compute_emotion_intensity_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.BoolTensor,
        loss_fn: str,
        num_classes: int = 3,
        weight: float = 1.0,
        class_weights: torch.Tensor | list[float] | None = None
    ) -> torch.Tensor:
    """Compute emotion intensity classification loss."""
    if not mask.any():
        return torch.zeros(size=(), device=y_pred.device)

    y_pred = y_pred[mask]
    y_true = y_true[mask]

    if isinstance(class_weights, list):
        class_weights = torch.tensor(class_weights, device=y_pred.device)

    if loss_fn == "cross_entropy":
        return weight * F.cross_entropy(y_pred, y_true, weight=class_weights)
    elif loss_fn == "focal":
        return weight * FocalLoss(gamma=2, alpha=class_weights, task_type='multi-class', num_classes=num_classes)(y_pred, y_true)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def va_loss(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    mask: torch.BoolTensor,
    weight: float = 1.0,
    loss_type: str = "ccc"
) -> torch.Tensor:
    """
    PyTorch loss functions for VA tasks.
    
    Args:
        preds: Predictions tensor (N, T, 1)
        targets: Targets tensor (N, T, 1)
        mask: Boolean mask for valid frames (N, T)
        loss_type: One of ['ccc', 'mse', 'ccc_mse']
    
    Returns:
        torch.Tensor: Loss value
    """
    if not mask.any():
        return torch.zeros(size=(), device=y_pred.device)

    y_pred = y_pred[mask]
    y_true = y_true[mask]

    if loss_type == "ccc":
        return weight * (1 - ccc_loss(y_pred, y_true))
    elif loss_type == "mse":
        return weight * torch.nn.functional.mse_loss(y_pred, y_true)
    elif loss_type == "ccc_mse":
        return weight * (0.5*(1 - ccc_loss(y_pred, y_true)) + \
               0.5*torch.nn.functional.mse_loss(y_pred.view(-1), y_true))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def ccc_loss(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Concordance Correlation Coefficient Loss"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    mean_pred = torch.mean(y_pred)
    mean_target = torch.mean(y_true)
    
    cov = torch.mean((y_pred - mean_pred) * (y_true - mean_target))
    var_pred = torch.var(y_pred, unbiased=False)
    var_target = torch.var(y_true, unbiased=False)
    
    ccc = (2 * cov) / (var_pred + var_target + (mean_pred - mean_target)**2 + 1e-8)
    return ccc


def consistency_loss(
        y_preds: list[torch.Tensor],
        tasks: dict[str, dict[str, Any]],
        consistency_rules: list[dict[str, Any]],
    ) -> torch.Tensor:
    total_loss = torch.zeros(size=(), device=y_preds[0].device)

    for rule in consistency_rules:
        if rule['name'] == 'sentiment':
            pred_cls = y_preds[list(tasks.keys()).index('sentiment_class')]
            pred_reg = y_preds[list(tasks.keys()).index('sentiment')]
            total_loss += compute_sentiment_consistency_loss(pred_cls, pred_reg, rule['weight'])

    return total_loss


def reconstruction_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.BoolTensor,
        weight: float = 1.0
    ) -> torch.Tensor:
    """Compute reconstruction loss."""
    if not mask.any():
        return torch.zeros(size=(), device=y_pred.device)

    y_pred = y_pred[mask]
    y_true = y_true[mask]
    return weight * torch.nn.functional.mse_loss(y_pred, y_true)


def multitarget_loss(
        y_preds: dict[str, torch.Tensor],
        y_targets: list[torch.Tensor],
        target_masks: list[torch.BoolTensor], 
        tasks: dict[str, dict[str, Any]],
        awl: AutomaticWeightedLoss | None = None
    ) -> torch.Tensor:
    """Compute multitarget loss with masking.

    Args:
        preds (list[torch.Tensor]): List of predictions for each task.
            [sentiment_cls (B, num_classes), sentiment_reg (B,),
             emotion_cls (B, n_classes_ec), intensity (B, n_classes_ei),
             valence (B, T), arousal (B, T)]
        targets (list[torch.Tensor]): Corresponding targets for each prediction.
        target_masks (list[torch.BoolTensor]): Masks indicating target availability.
        tasks (dict[str, dict[str, Any]]): Task configuration dictionary.
            Each task should specify:
            - type: str ('sentiment_cls', 'sentiment_reg', 'emotion', 'intensity', 'va')
            - num_classes: int (for classification tasks)
            - is_framewise: bool (for emotion classification)
            - weight: float (loss weight)
            - consistency_weight: float (for sentiment task)
        awl (AutomaticWeightedLoss | None): Automatic weighted loss nn.Module

    Returns:
        torch.Tensor: Total weighted loss (scalar).
    """
    total_loss = [] # torch.zeros(size=(), device=y_targets[0].device)

    for task_ind, (task_name, task_info) in enumerate(tasks.items()):

        if task_name == 'emotion_class':
            y_pred = y_preds[task_name]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss.append(compute_emotion_class_loss(
                y_pred=y_pred, 
                y_true=y_target, 
                mask=mask, 
                loss_fn=task_info['loss_fn'],
                num_classes=task_info['num_classes'],
                weight=task_info.get('weight', 1.0), 
                is_framewise=False, 
                class_weights=task_info.get('class_weights', None)
            ))
        
        elif task_name == 'emotion_class_fw':
            y_pred = y_preds[task_name]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss.append(compute_emotion_class_loss(
                y_pred=y_pred, 
                y_true=y_target, 
                mask=mask, 
                loss_fn=task_info['loss_fn'],
                num_classes=task_info['num_classes'],
                weight=task_info.get('weight', 1.0), 
                is_framewise=True,
                class_weights=task_info.get('class_weights', None)
            ))

        elif task_name == 'emotion_intensity':
            y_pred = y_preds[task_name]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss.append(compute_emotion_intensity_loss(
                y_pred=y_pred, 
                y_true=y_target, 
                mask=mask, 
                loss_fn=task_info['loss_fn'],
                num_classes=task_info['num_classes'],
                weight=task_info.get('weight', 1.0)
            ))

        elif task_name == 'sentiment':
            y_pred = y_preds[task_name]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss.append(compute_sentiment_loss(
                y_pred=y_pred, 
                y_true=y_target, 
                mask=mask,
                weight=task_info.get('weight', 1.0)
            ))

        elif task_name == 'sentiment_class':
            y_pred = y_preds[task_name]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss.append(compute_sentiment_class_loss(
                y_pred=y_pred, 
                y_true=y_target, 
                mask=mask, 
                loss_fn=task_info['loss_fn'],
                num_classes=task_info['num_classes'],
                weight=task_info.get('weight', 1.0)
            ))

        elif task_name == 'valence':
            y_pred = y_preds[task_name]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss.append(va_loss(
                y_pred=y_pred, 
                y_true=y_target, 
                mask=mask,
                weight=task_info.get('weight', 1.0),
                loss_type="ccc"
            ))

        elif task_name == 'arousal':
            y_pred = y_preds[task_name]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss.append(va_loss(
                y_pred=y_pred, 
                y_true=y_target, 
                mask=mask,
                weight=task_info.get('weight', 1.0),
                loss_type="ccc"
            ))

        elif task_name in ['tmm_wavlm_baseplus', 'tmm_clip', 'tmm_xml_roberta']:
            y_pred = y_preds[task_name]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss.append(reconstruction_loss(
                y_pred=y_pred, 
                y_true=y_target, 
                mask=mask,
                weight=task_info.get('weight', 1.0)
            ))

    if awl is not None:
        total_loss = awl(total_loss)
    else:
        total_loss = sum(total_loss)

    return total_loss


if __name__ == '__main__':

    awl = AutomaticWeightedLoss(2)
    print('awl parameters:', awl.parameters())

    sentiment_class_pred = torch.randn(16, 3).to(device='cuda:0')
    sentiment_class_target = torch.randint(0, 3, (16,)).to(device='cuda:0')
    sentiment_class_mask = torch.randint(0, 2, (16,)).bool().to(device='cuda:0')

    sentiment_class_loss = compute_sentiment_class_loss(
        sentiment_class_pred, 
        sentiment_class_target, 
        sentiment_class_mask, 
        loss_fn="cross_entropy",
        num_classes=3,
        weight=1.0
    )

    sentiment_class_loss = compute_sentiment_class_loss(
        sentiment_class_pred, 
        sentiment_class_target, 
        sentiment_class_mask, 
        loss_fn="focal",
        num_classes=3,
        weight=1.0,
        class_weights=torch.tensor([0.1, 0.2, 0.7])
    )