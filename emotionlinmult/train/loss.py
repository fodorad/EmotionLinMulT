import torch
import torch.nn.functional as F
from typing import Any
from exordium.utils.loss import bell_l2_l1_loss


def compute_sentiment_class_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.BoolTensor,
        num_classes: int = 3,
        weight: float = 1.0
    ) -> torch.Tensor:
    """Compute sentiment classification loss."""
    if not mask.any():
        return torch.zeros(size=(), device=y_pred.device)

    y_pred = y_pred[mask]
    y_true = y_true[mask]
    return weight * F.cross_entropy(y_pred, y_true)


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
    return weight * bell_l2_l1_loss(y_pred, y_true)


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


def compute_emotion_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.BoolTensor,
        num_classes: int,
        weight: float = 1.0,
        is_framewise: bool = False
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

    return weight * F.cross_entropy(y_pred, y_true)


def compute_intensity_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.BoolTensor,
        num_classes: int,
        weight: float = 1.0,
        is_framewise: bool = False
    ) -> torch.Tensor:
    """Compute emotion intensity classification loss."""
    if not mask.any():
        return torch.zeros(size=(), device=y_pred.device)
        
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    return weight * F.cross_entropy(y_pred, y_true)


def compute_valence_loss(
        y_pred_v: torch.Tensor,
        y_true_v: torch.Tensor,
        mask_v: torch.BoolTensor,
        weight: float = 1.0
    ) -> torch.Tensor:
    """Compute valence regression loss."""
    loss = torch.zeros(size=(), device=y_pred_v.device)

    if mask_v.any():
        loss += bell_l2_l1_loss(y_pred_v[mask_v], y_true_v[mask_v])

    return weight * loss


def compute_arousal_loss(
        y_pred_a: torch.Tensor,
        y_true_a: torch.Tensor,
        mask_a: torch.BoolTensor,
        weight: float = 1.0
    ) -> torch.Tensor:
    """Compute arousal regression loss."""
    loss = torch.zeros(size=(), device=y_pred_a.device)

    if mask_a.any():
        loss += bell_l2_l1_loss(y_pred_a[mask_a], y_true_a[mask_a])

    return weight * loss


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


def multitarget_loss(
        y_preds: list[torch.Tensor],
        y_targets: list[torch.Tensor],
        target_masks: list[torch.BoolTensor], 
        tasks: dict[str, dict[str, Any]]
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

    Returns:
        torch.Tensor: Total weighted loss (scalar).
    """
    total_loss = torch.zeros(size=(), device=y_preds[0].device)

    for task_ind, (task_name, task_info) in enumerate(tasks.items()):

        if task_name == 'sentiment_class':
            y_pred = y_preds[task_ind]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss += compute_sentiment_class_loss(
                y_pred, y_target, mask,
                task_info['num_classes'],
                task_info['weight']
            )

        elif task_name == 'sentiment':
            y_pred = y_preds[task_ind]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss += compute_sentiment_loss(
                y_pred, y_target, mask,
                task_info['weight']
            )

        elif task_name == 'emotion':
            y_pred = y_preds[task_ind]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss += compute_emotion_loss(
                y_pred, y_target, mask, task_info['num_classes'],
                task_info['weight'], is_framewise=task_info.get('is_framewise', False)
            )

        elif task_name == 'intensity':
            y_pred = y_preds[task_ind]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss += compute_intensity_loss(
                y_pred, y_target, mask,
                task_info['num_classes'],
                task_info['weight']
            )

        elif task_name == 'valence':
            y_pred = y_preds[task_ind]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss += compute_valence_loss(
                y_pred, y_target, mask,
                task_info['weight']
            )

        elif task_name == 'arousal':
            y_pred = y_preds[task_ind]
            y_target = y_targets[task_ind]
            mask = target_masks[task_ind]
            total_loss += compute_arousal_loss(
                y_pred, y_target, mask,
                task_info['weight']
            )

    return total_loss


if __name__ == '__main__':
    
    sentiment_class_pred = torch.randn(16, 3)
    sentiment_class_target = torch.randint(0, 3, (16,))
    sentiment_class_mask = torch.randint(0, 2, (16,)).bool()
    
    sentiment_pred = torch.randn(16)
    sentiment_target = torch.randn(16)
    sentiment_mask = torch.randint(0, 2, (16,)).bool()
    
    sentiment_class_loss = compute_sentiment_class_loss(
        sentiment_class_pred, sentiment_class_target, sentiment_class_mask, 3, 1.0
    )
    sentiment_loss = compute_sentiment_loss(
        sentiment_pred, sentiment_target, sentiment_mask, 1.0
    )
    sentiment_consistency_loss = compute_sentiment_consistency_loss(
        sentiment_class_pred, sentiment_pred, sentiment_mask, 1.0
    )

    print(sentiment_class_loss.item())
    print(sentiment_loss.item())
    print(sentiment_consistency_loss.item())