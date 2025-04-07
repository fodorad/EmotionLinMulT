import torch
import numpy as np


def get_scheduler(optimizer, config: dict):
    lr_scheduler_config = config.get('lr_scheduler', {})
    optimizer_config = config.get('optimizer', {})
    base_lr = optimizer_config.get('base_lr', 1e-3)
    name = lr_scheduler_config.get('name', 'reducelronplateau')

    if name in ['cosine_warmup', 'warmup']:

        total_steps = int(config.get('total_steps',
            config.get('n_epochs') * config.get('train_size') / config.get('batch_size')))
        warmup_steps = int(total_steps * lr_scheduler_config.get('warmup_steps', 0.05))

        if name == 'cosine_warmup':
            scheduler = CosineAnnealingWarmupScheduler(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                max_lr=base_lr,
                total_steps=total_steps
            )
        else:
            scheduler = WarmupScheduler(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                base_lr=base_lr
            )

    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_scheduler_config.get('factor', 0.1),
            patience=lr_scheduler_config.get('patience', 5),
            min_lr=1e-6
        )

    return scheduler


class WarmupScheduler:

    def __init__(self, optimizer, warmup_steps, base_lr):
        """
        Warmup scheduler to increase learning rate linearly during warmup.
        
        Args:
            optimizer: The optimizer being used.
            warmup_steps: Number of steps to warm up the learning rate.
            base_lr: Base learning rate to reach after warmup.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            lr = (self.current_step / float(self.warmup_steps)) * self.base_lr
        else:
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class CosineAnnealingWarmupScheduler:

    def __init__(self, optimizer, warmup_steps, max_lr, total_steps):
        """
        Scheduler that combines warmup with cosine annealing.
        
        Args:
            optimizer: The optimizer being used.
            warmup_steps: Number of steps to warm up the learning rate.
            max_lr: Maximum learning rate (after warmup).
            total_steps: Total number of training steps.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.current_step = 0
        self.cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps - warmup_steps)
    
    def step(self, val_loss=None):
        if self.current_step < self.warmup_steps:
            lr = (self.current_step / float(self.warmup_steps)) * self.max_lr
        else:
            self.cosine_annealing_scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'cosine_scheduler_state': self.cosine_annealing_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.cosine_annealing_scheduler.load_state_dict(state_dict['cosine_scheduler_state'])

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor