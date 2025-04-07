from typing import Callable
import torch
import torch.nn as nn
from emotionlinmult.train.metrics import CCC


LOSS_FNS = {
    'cross_entropy': nn.CrossEntropyLoss()
}


def multitarget_loss(
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        target_masks: list[torch.Tensor],
        loss_fn_names: list[str],
        shake_coeffs: torch.Tensor,
    ):
    """Compute multitarget loss with masking.

    Args:
        preds (list[torch.Tensor]): List of predictions [(B, n_classes), (B,), (B, T), (B, T)].
        targets (list[torch.Tensor]): List of targets [(B,), (B,), (B, T), (B, T)].
        target_masks (list[torch.BoolTensor]): List of masks [(B,), (B,), (B, T), (B, T)] indicating target availability.

    Returns:
        torch.Tensor: Total loss (scalar).
    """
    # Emotion class (classification, (B, n_classes_ec))
    # Emotion intensity (classification, (B, n_classes_ei))
    # Valence Arousal (regression, (B, 2))
    losses = []
    for index, loss_fn_name in enumerate(loss_fn_names):
        if target_masks[index].any(): # Check if any sample has this target available
            pred = preds[index][target_masks[index]] # Use only valid samples
            target = targets[index][target_masks[index]]
            loss_value = LOSS_FNS[loss_fn_name](pred, target.long())
        else:
            loss_value = torch.zeros(size=(), device=preds[0].device)
        losses.append(loss_value)

    # Valence loss (regression, (B, T)) and Arousal loss (regression, (B, T))
    #if target_masks[2].any() or target_masks[3].any():
    #    ccc_valence = CCC(ground_truth=targets[2], predictions=preds[2], mask=target_masks[2])
    #    ccc_arousal = CCC(ground_truth=targets[3], predictions=preds[3], mask=target_masks[3])
    #    va_loss_value = 1 - 0.5 * (ccc_valence + ccc_arousal)
    #else:
    #    va_loss_value = torch.zeros(size=(), device=preds[2].device)

    loss_values_tensor = torch.stack(losses)
    shake_coeff_tensor = shake_coeffs.to(loss_values_tensor[0].device) # Ensure it’s on the same device
    normalized_shake_coeffs = shake_coeff_tensor / torch.sum(shake_coeff_tensor)

    combined_loss = torch.sum(normalized_shake_coeffs * loss_values_tensor)
    #combined_loss = torch.sum(loss_values_tensor)
    return combined_loss


def mse_loss_va(x, y, clamp=False):

    if clamp:
        val_pred, val_true = x[:, 0].clamp(-1, 1), y[:, 0].clamp(-1, 1)
        arsl_pred, arsl_true = x[:, 1].clamp(-1, 1), y[:, 1].clamp(-1, 1)
    else:
        val_pred, val_true = x[:, 0], y[:, 0]
        arsl_pred, arsl_true = x[:, 1], y[:, 1]

    mse_v = torch.nn.functional.mse_loss(val_pred, val_true)
    mse_a = torch.nn.functional.mse_loss(arsl_pred, arsl_true)
    loss = mse_v + mse_a
    return loss


def ccc(x, y):
    pcc = torch.corrcoef(torch.stack((x, y), dim=0))[0, 1]
    num = 2 * pcc * x.std() * y.std()
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    ccc = num / den
    return torch.nan_to_num(ccc, nan=0)


def _ccc_loss_va(x, y, clamp=False):
    # x and y shape: (bs, 2)
    # first dimension for valence, second for arousal

    if clamp:
        val_pred, val_true = x[:, 0].clamp(-1, 1), y[:, 0].clamp(-1, 1)
        arsl_pred, arsl_true = x[:, 1].clamp(-1, 1), y[:, 1].clamp(-1, 1)
    else:
        val_pred, val_true = x[:, 0], y[:, 0]
        arsl_pred, arsl_true = x[:, 1], y[:, 1]

    ccc_v = ccc(val_pred, val_true)
    ccc_a = ccc(arsl_pred, arsl_true)

    loss = 1 - 0.5 * (ccc_v + ccc_a)

    return loss


def mse_ccc_loss_va(x, y, weights=(1, 1), clamp=False):
    loss = (weights[0] * mse_loss_va(x, y, clamp)) +\
           (weights[1] * ccc_loss_va(x, y, clamp))
    return loss


def dyn_wt_mse_ccc_loss_va(x, y, epoch, max_epochs, alpha=1, weight_exponent=2, clamp=False):

    weights = (alpha * ((epoch/max_epochs)**weight_exponent), 1.0 - ((epoch/max_epochs)**weight_exponent))
    loss = (weights[0] * mse_loss_va(x, y, clamp)) +\
           (weights[1] * ccc_loss_va(x, y, clamp))

    return loss


def dyn_wt_mse_ccc_loss(x, y, epoch, max_epochs, alpha=1, weight_exponent=2, clamp=False):

    weights = (alpha * ((epoch/max_epochs)**weight_exponent), 1.0 - ((epoch/max_epochs)**weight_exponent))

    if clamp:
        x, y = x.clamp(-1, 1), y.clamp(-1, 1)

    loss = (weights[0] * torch.nn.functional.mse_loss(x, y)) +\
           (weights[1] * (1.0 - ccc(x, y)))

    return loss