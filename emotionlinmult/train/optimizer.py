import torch


def get_optimizer(params, config: dict):
    optimizer_config = config.get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'adam')
    base_lr = optimizer_config.get('base_lr', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 1e-5)

    if optimizer_name == 'radam':
        optimizer = torch.optim.RAdam(
            params,
            lr=base_lr,
            weight_decay=float(weight_decay),
            decoupled_weight_decay=optimizer_config.get('decoupled_weight_decay', True)
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=float(weight_decay)
        )
    else:
        optimizer = torch.optim.Adam(params, lr=base_lr)

    return optimizer