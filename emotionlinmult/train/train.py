import os
import json
import argparse
import yaml
from pprint import pprint
from pathlib import Path
from datetime import datetime
from typing import Callable
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import seed_everything

from linmult import LinMulT, LinT, load_config
from emotionlinmult.train.parser import argparser
from emotionlinmult.train.history import History
from emotionlinmult.train.loss import multitarget_loss, consistency_loss
from emotionlinmult.train.metrics import calculate_sentiment, calculate_sentiment_class
from emotionlinmult.train.datamodule import MultiDatasetModule


class ModelWrapper(L.LightningModule):

    def __init__(self, model, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters('config')

        self.model = model
        self.tasks = config['tasks']
        self.consistency_rules = config['consistency_rules']

        self.train_preds = []
        self.train_targets = []
        self.train_masks = []

        self.valid_preds = []
        self.valid_targets = []
        self.valid_masks = []

        self.test_preds = []
        self.test_targets = []
        self.test_masks = []

        self.test_info = []

        self.log_dir = Path(config['experiment_dir'])
        self.history = History(self.log_dir)


    def forward(self, x, masks=None):
        return self.model(x, masks)


    def training_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']] # list of shapes [(B, T, F), ...]
        x_masks = [batch[f'{feature_name}_mask'] for feature_name in self.config['feature_list']] # list of shapes [(B, T), ...]
        y_true = [batch[task] for task in self.tasks] # list of shapes [(B,) or (B, T), ...]. sentiment: (B,), sentiment_class: (B,)
        y_true_masks = [batch[f'{task}_mask'] for task in self.tasks] # list of shapes [(B,) or (B, T), ...]

        preds_heads = self(x, x_masks) # list of output heads. sentiment: (B, 1), sentiment_class: (B, 3)
        loss = multitarget_loss(preds_heads, y_true, y_true_masks, self.tasks)

        if self.consistency_rules:
            loss += consistency_loss(preds_heads, self.tasks, self.consistency_rules)

        self.log('train_loss', loss, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=self.config['batch_size'])
        self.train_preds.append(preds_heads)
        self.train_targets.append(y_true)
        self.train_masks.append(y_true_masks)

        return loss


    def on_train_epoch_end(self):
        # training_step -> validation_step -> on_valid_epoch_end -> on_train_epoch_end
        pred_tasks = [torch.cat([batch[task_ind] for batch in self.train_preds], dim=0) for task_ind in range(len(self.tasks))]
        target_tasks = [torch.cat([batch[task_ind] for batch in self.train_targets], dim=0) for task_ind in range(len(self.tasks))]
        mask_tasks = [torch.cat([batch[task_ind] for batch in self.train_masks], dim=0) for task_ind in range(len(self.tasks))]

        for task_ind, (task_name, task_info) in enumerate(self.tasks.items()):
            masks = mask_tasks[task_ind] # Shape (N,)
            preds = pred_tasks[task_ind][masks] # Shape (M,) where M <= N
            targets = target_tasks[task_ind][masks] # Shape (M,) where M <= N

            if task_name == 'sentiment':
                metrics = calculate_sentiment(preds, targets)
            elif task_name == 'sentiment_class':
                metrics = calculate_sentiment_class(preds, targets)
            else:
                raise ValueError(f"Unknown task: {task_name}")

            for metric_name, metric_value in metrics.items():
                self.history.update(phase="train", task=task_name, metric=metric_name, value=metric_value, epoch=self.current_epoch)

                if metric_name in task_info['metrics']:
                    self.log(f'train_{task_name}_{metric_name}', metric_value, prog_bar=False, logger=True, on_epoch=True)
                    self.history.plot(task_name, metric_name)

            if task_name == 'sentiment_class':
                self.history.plot_ncm('valid', task_name, 'F1', task_info['num_classes'])
                self.history.plot_ncm('train', task_name, 'F1', task_info['num_classes'])

        avg_loss = self.trainer.logged_metrics['train_loss']
        self.history.update(phase="train", task="all", metric="avg_loss", value=avg_loss.item(), epoch=self.current_epoch)
        self.log('train_loss', avg_loss.item(), prog_bar=True, logger=True, on_epoch=True, batch_size=self.config['batch_size'])
        self.history.plot('all', 'avg_loss')
        self.history.save()

        self.train_preds = []
        self.train_targets = []
        self.train_masks = []


    def validation_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']] # list of shapes [(B, T, F), ...]
        x_masks = [batch[f'{feature_name}_mask'] for feature_name in self.config['feature_list']] # list of shapes [(B, T), ...]
        y = [batch[task] for task in self.tasks] # list of shapes [(B,) or (B,T), ...]
        y_masks = [batch[f'{task}_mask'] for task in self.tasks] # list of shapes [(B, T), ...]

        preds_heads = self(x, x_masks) # list of output heads. sentiment: (B, 1), sentiment_class: (B, 3)
        loss = multitarget_loss(preds_heads, y, y_masks, self.tasks)
        
        if self.consistency_rules:
            loss += consistency_loss(preds_heads, self.tasks, self.consistency_rules)

        self.log('valid_loss', loss, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config['batch_size'])
        self.valid_preds.append(preds_heads)
        self.valid_targets.append(y)
        self.valid_masks.append(y_masks)

        return loss


    def on_validation_epoch_end(self):
        preds_tasks = [torch.cat([batch[task_ind] for batch in self.valid_preds], dim=0) for task_ind in range(len(self.tasks))]
        targets_tasks = [torch.cat([batch[task_ind] for batch in self.valid_targets], dim=0) for task_ind in range(len(self.tasks))]
        masks_tasks = [torch.cat([batch[task_ind] for batch in self.valid_masks], dim=0) for task_ind in range(len(self.tasks))]

        for task_ind, (task_name, task_info) in enumerate(self.tasks.items()):
            masks = masks_tasks[task_ind] # Shape (N,)
            preds = preds_tasks[task_ind][masks] # Shape (M,) where M <= N
            targets = targets_tasks[task_ind][masks] # Shape (M,) where M <= N

            if task_name == 'sentiment':
                metrics = calculate_sentiment(preds, targets)
            elif task_name == 'sentiment_class':
                metrics = calculate_sentiment_class(preds, targets)
            else:
                raise ValueError(f"Unknown task: {task_name}")

            for metric_name, metric_value in metrics.items():
                self.history.update(phase="valid", task=task_name, metric=metric_name, value=metric_value, epoch=self.current_epoch)

                if metric_name in task_info['metrics']:
                    self.log(f'valid_{task_name}_{metric_name}', metric_value, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config['batch_size'])

        avg_loss = self.trainer.logged_metrics['valid_loss']
        self.history.update(phase="valid", task="all", metric="avg_loss", value=avg_loss.item(), epoch=self.current_epoch)
        self.log('valid_loss', avg_loss.item(), prog_bar=True, logger=True, on_epoch=True, batch_size=self.config['batch_size'])

        self.valid_preds = []
        self.valid_targets = []
        self.valid_masks = []


    def test_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']] # list of shapes [(B, T, F), ...]
        x_masks = [batch[f'{feature_name}_mask'] for feature_name in self.config['feature_list']] # list of shapes [(B, T), ...]

        preds_heads = self(x, x_masks) # list of heads

        y = [batch[task] for task in self.tasks] # list of shapes [(B,) or (B, T), ...]
        y_masks = [batch[f'{task}_mask'] for task in self.tasks] # list of shapes [(B,) or (B, T), ...]
 
        self.test_preds.append(preds_heads)
        self.test_targets.append(y)
        self.test_masks.append(y_masks)

        self.test_info.append(list(zip(batch['dataset'], batch['__key__']))) # list of (dataset, sample_key) pairs


    def on_test_epoch_end(self):
        pred_tasks = [torch.cat([batch[task_ind] for batch in self.test_preds], dim=0) for task_ind in range(len(self.tasks))]
        target_tasks = [torch.cat([batch[task_ind] for batch in self.test_targets], dim=0) for task_ind in range(len(self.tasks))]
        mask_tasks = [torch.cat([batch[task_ind] for batch in self.test_masks], dim=0) for task_ind in range(len(self.tasks))]
        info = [item for sublist in self.test_info for item in sublist]

        # Calculate metrics for each task
        for task_ind, (task_name, task_info) in enumerate(self.tasks.items()):
            masks = mask_tasks[task_ind] # Shape (N,)
            preds = pred_tasks[task_ind][masks] # Shape (M,) where M <= N
            targets = target_tasks[task_ind][masks] # Shape (M,) where M <= N

            if task_name == 'sentiment':
                metrics = calculate_sentiment(preds, targets)
            elif task_name == 'sentiment_class':
                metrics = calculate_sentiment_class(preds, targets)
            else:
                raise ValueError(f"Unknown task: {task_name}")

            for metric_name, metric_value in metrics.items():
                self.history.update(phase="test", task=task_name, metric=metric_name, value=metric_value, epoch=self.current_epoch)

                if metric_name in task_info['metrics']:
                    self.log(f'test_{task_name}_{metric_name}', metric_value, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config['batch_size'])

        dataset_predictions = {}
        # Save predictions and targets for each sample
        for task_ind, (task_name, task_info) in enumerate(self.tasks.items()):
            preds = pred_tasks[task_ind] # Shape (M,) or (M, T) or (M, T, C)
            targets = target_tasks[task_ind] # Shape (M,) or (M, T)
            masks = mask_tasks[task_ind] # Shape (N,) or (N, T)

            for sample_ind, (dataset, sample_key) in enumerate(info):
                # Handle different mask types based on task
                if task_name in ['sentiment', 'sentiment_class', 'emotion_class', 'intensity']:
                    # Scalar mask (B,)
                    if not masks[sample_ind]:
                        continue  # Skip if sample is masked
                else:  # valence, arousal, emotion_class_fw
                    # Frame-wise mask (B, T)
                    if not masks[sample_ind].any():
                        continue  # Skip if all frames are masked

                # Initialize dataset and task dictionaries if they don't exist
                if dataset not in dataset_predictions:
                    dataset_predictions[dataset] = {}
                if task_name not in dataset_predictions[dataset]:
                    dataset_predictions[dataset][task_name] = {}
                if sample_key not in dataset_predictions[dataset][task_name]:
                    dataset_predictions[dataset][task_name][sample_key] = {}
                
                # Convert predictions and targets to Python types for JSON serialization
                if task_name == 'sentiment': # save the value
                    pred_value = float(preds[sample_ind].item())
                    target_value = float(targets[sample_ind].item())
                elif task_name in ['sentiment_class', 'emotion_class', 'intensity']: # save all logits
                    pred_value = [float(x) for x in preds[sample_ind].tolist()]
                    target_value = int(targets[sample_ind].item())
                elif task_name in ['valence', 'arousal', 'emotion_class_fw']: # frame-wise prediction for a single sample is (T, 1) or (T, C)
                    # Get frame-wise predictions and targets
                    if task_name in ['valence', 'arousal']:
                        # For valence/arousal, shape is (T, 1)
                        pred_frames = preds[sample_ind].squeeze(-1)  # (T,)
                        target_frames = targets[sample_ind].squeeze(-1)  # (T,)
                        frame_masks = masks[sample_ind]  # (T,)
                        
                        # Create frame-wise entries only for valid frames
                        pred_value = {}
                        target_value = {}
                        for frame_id in range(len(pred_frames)):
                            if frame_masks[frame_id]:
                                pred_value[str(frame_id)] = float(pred_frames[frame_id].item())
                                target_value[str(frame_id)] = float(target_frames[frame_id].item())
                                
                    elif task_name == 'emotion_class_fw':
                            # For emotion_class_fw, shape is (T, C)
                            pred_frames = preds[sample_ind]  # (T, C)
                            target_frames = targets[sample_ind]  # (T,)
                            frame_masks = masks[sample_ind]  # (T,)
                            
                            # Create frame-wise entries with logits only for valid frames
                            pred_value = {}
                            target_value = {}
                            for frame_id in range(len(pred_frames)):
                                if frame_masks[frame_id]:
                                    pred_value[str(frame_id)] = [float(x) for x in pred_frames[frame_id].tolist()]
                                    target_value[str(frame_id)] = int(target_frames[frame_id].item())
                    else:
                        raise ValueError(f"Unknown task: {task_name}")

                dataset_predictions[dataset][task_name][sample_key] = {
                    'y_pred': pred_value,
                    'y_true': target_value
                }

        # Save predictions for each dataset
        predictions_dir = Path(self.config['experiment_dir']) / 'predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        for dataset, predictions in dataset_predictions.items():
            output_path = predictions_dir / f'test_{dataset.lower()}.json'
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            if task_name == 'sentiment_class':
                self.history.plot_ncm('test', task_name, 'F1', task_info['num_classes'])

        self.history.save_test()

        self.test_preds = []
        self.test_targets = []
        self.test_masks = []
        self.test_info = []


    def configure_optimizers(self):
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam')
        base_lr = float(optimizer_config.get('base_lr', 1e-3))
        weight_decay = float(optimizer_config.get('weight_decay', 0))

        # Configure optimizer
        if optimizer_name == 'radam':
            optimizer = torch.optim.RAdam(
                self.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
                decoupled_weight_decay=optimizer_config.get('decoupled_weight_decay', False)
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=base_lr,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=base_lr)

        # Configure learning rate scheduler
        lr_scheduler_config = self.config.get('lr_scheduler', {})
        lr_scheduler_name = lr_scheduler_config.get('name', 'ReduceLROnPlateau')
        

        # Configure warmup scheduler
        if config.get('warmup_epochs', 0) > 0:
            # write a warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=config['warmup_epochs']
            )
        else:
            warmup_scheduler = None

        if lr_scheduler_name == 'ReduceLROnPlateau':
            main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(lr_scheduler_config.get('factor', 0.1)),
                patience=int(lr_scheduler_config.get('patience', 5)),
                min_lr=1e-8
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": main_scheduler,
                    "monitor": lr_scheduler_config.get('monitor', 'valid_loss')
                }
            }

        if lr_scheduler_name == 'OneCycleLR':
            if self.trainer.estimated_stepping_batches == -1: # webdataset do not have len
                n_batch = self.config.get('n_batch', 0.)
                total_steps = 0
                if n_batch != 0:
                    total_steps = n_batch * self.config["n_epochs"]
                    total_steps //= (self.trainer.accumulate_grad_batches * max(1, self.trainer.num_devices))
            else:
                total_steps = self.trainer.estimated_stepping_batches
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=base_lr, total_steps=total_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }

        # Main scheduler
        if lr_scheduler_name == 'CosineAnnealingLR':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['n_epochs']-config.get('warmup_epochs', 0))
        else:
            raise ValueError(f'Given lr scheduler is not supported: {lr_scheduler_name}')

        # Combine warmup and main scheduler using SequentialLR
        if config.get('warmup_epochs', 0) > 0:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config['warmup_epochs']]
            )
        else:
            lr_scheduler = main_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


def train_model(config: dict):
    seed_everything(config.get('seed', 42))
    torch.set_float32_matmul_precision(config.get('float32_matmul_precision', 'medium')) # medium, high, highest
    experiment_name = config.get('experiment_name', datetime.now().strftime("%Y%m%d-%H%M%S"))
    experiment_dir = Path(config.get('output_dir', 'results')) / experiment_name
    if experiment_dir.exists() and 'test_only' not in config:
        raise ValueError(f'Experiment with {experiment_name} already exists.')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config['experiment_dir'] = str(experiment_dir)

    # Initialize the DataModule
    data_module = MultiDatasetModule(config=config)

    # Initialize the model
    if config['model_name'] == 'LinT':
        model = LinT(config=config)
    elif config['model_name'] == 'LinMulT':
        model = LinMulT(config=config)
    else:
        raise ValueError(f'Unsupported model name {config["model_name"]}')

    lightning_model = ModelWrapper(model, config=config)

    # Define the callbacks and logger
    callbacks = []
    for checkpoint_config in config['checkpoints']:
        callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=experiment_dir / 'checkpoint',
            filename=f"{checkpoint_config['name']}",
            monitor=checkpoint_config['monitor'],
            mode=checkpoint_config['mode'],
            save_top_k=1,
            verbose=True,
            save_weights_only=True
        )
        callbacks.append(callback)

    config_es = config.get('early_stopping', False)
    if config_es:
        early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor=config_es['monitor'], # 'valid_loss'
            patience=config_es['patience'], # 5
            mode=config_es['mode'], # 'min'
            verbose=True
        )
        callbacks.append(early_stopping)

    csv_logger = L.pytorch.loggers.CSVLogger(save_dir=str(experiment_dir), name="csv_logs")

    # Define the trainer
    trainer = L.Trainer(
        accelerator=config.get('accelerator', 'gpu'),
        devices=config.get('devices', [0]),
        max_epochs=config.get('n_epochs', 100),
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=csv_logger,
        num_sanity_val_steps=0,
        #limit_train_batches=1,
        #limit_val_batches=1,
        #limit_test_batches=1,
        gradient_clip_val=1.0
    )

    # Train the model
    if 'test_only' not in config:
        if 'cp_path' in config:
            trainer.fit(lightning_model, datamodule=data_module, ckpt_path=config['cp_path'])
        else:
            trainer.fit(lightning_model, datamodule=data_module)

    # Test the model on the test set
    print("Evaluating on the test set...")
    if 'cp_path' in config:
        checkpoint_path = config['cp_path']
        print(f"Loading model from: {checkpoint_path}")
    else:
        checkpoint_path = callbacks[1].best_model_path # checkpoint_valid_mae
        print(f"Loading best model from: {checkpoint_path}")

    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=checkpoint_path, map_location=torch.device(f'cuda:{config.get("gpu_id", 0)}'))
    test_results = trainer.test(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    config: dict = argparser()
    train_model(config)