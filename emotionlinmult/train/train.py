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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.functional import mean_absolute_error
from exordium.utils.loss import bell_l2_l1_loss
from linmult import LinMulT, LinT, apply_logit_aggregation, load_config
from emotionlinmult.train.datamodule import EmotionDataModule
from emotionlinmult.train.metrics import classification_metrics, va_metrics
from emotionlinmult.train.loss import multitarget_loss
from emotionlinmult.train.optimizer import get_optimizer
from emotionlinmult.train.scheduler import get_scheduler
from emotionlinmult.train.history import History


class ModelWrapper(L.LightningModule):

    def __init__(self, model, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters('config')

        self.model = model

        self.tasks = list(config['tasks'].keys())
        self.loss_fns = [config['tasks'][task]['loss_fn'] for task in self.tasks]
        self.datasets = list(config['datasets'].keys())
        self.history = History()

        self.train_preds = []
        self.train_targets = []
        self.train_masks = []

        self.valid_preds = []
        self.valid_targets = []
        self.valid_masks = []

        self.test_preds = []
        self.test_targets = []
        self.test_masks = []

        self.shake = config.get('shake', False)
        self.shake_coeffs = torch.rand(len(self.loss_fns)) if self.shake else torch.ones(len(self.loss_fns))


    def forward(self, x, masks=None, names=None):
        return self.model(x, masks, names) # LinMulT or LinT


    def training_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']] # list of shapes [(B,T,F) or (B,L,T,F), ...]
        x_masks = [batch[f'{feature_name}_mask'] for feature_name in self.config['feature_list']] # list of shapes [(B,T), ...]
        x_mask = batch[f'{self.config["feature_list"][-1]}_mask']

        preds_heads = self(x, x_masks, names=self.config['feature_list']) # list of shapes [(B, T, n_classes_ec), (B, T, n_classes_ei), (B, T, 2)]  - emotion class, emotion intensity, valence & arousal

        #emotion_class = apply_logit_aggregation(preds_heads[0], mask=x_mask, method='meanpooling') # (B, T, n_classes) -> (B, n_classes)
        #emotion_intensity = apply_logit_aggregation(preds_heads[1], mask=x_mask, method='meanpooling') # (B, T, 1) -> (B, 1)
        #valence = preds_heads[2][:,:,0]
        #arousal = preds_heads[2][:,:,1]
        #preds = []
        #preds.append(emotion_class) # emotion class (B, n_classes_ec)
        #preds.append(emotion_intensity) # emotion intensity (B, n_classes_ei)
        #preds.append(valence) # valence (B, T)
        #preds.append(arousal) # arousal (B, T)

        y = [batch[task] for task in self.tasks] # list of shapes [(B,) or (B,T), ...]
        y_masks = [batch[f'{task}_mask'] for task in self.tasks] # list of shapes [(B, T), ...]

        loss = multitarget_loss(
            preds_heads, y, y_masks,
            self.loss_fns,
            self.shake_coeffs
        )

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.train_preds.append(preds_heads)
        self.train_targets.append(y)
        self.train_masks.append(y_masks)

        return loss


    def on_train_epoch_end(self):
        # training_step -> validation_step -> on_valid_epoch_end -> on_train_epoch_end
        preds = [torch.cat([batch[i] for batch in self.train_preds], dim=0) for i in range(len(self.tasks))]
        targets = [torch.cat([batch[i] for batch in self.train_targets], dim=0) for i in range(len(self.tasks))]
        masks = [torch.cat([batch[i] for batch in self.train_masks], dim=0) for i in range(len(self.tasks))]

        log_dir = Path(self.trainer.log_dir).parents[1] / 'visualization'
        for index, task in enumerate(self.tasks):
            task_info = self.config['tasks'][task]

            if task_info['type'] == 'classification':
                metrics = classification_metrics(y_true=targets[index], y_pred=preds[index], y_mask=masks[index], n_classes=task_info['num_classes'])
                for metric_name, metric_value in metrics.items():
                    self.history.update(phase="train", task=task, metric=metric_name, value=metric_value, epoch=self.current_epoch)

                    if metric_name in task_info['metrics']:
                        self.log(f'train_{task}_{metric_name}', metric_value, prog_bar=True, logger=True, on_epoch=True)
                        self.history.plot(task, metric_name, log_dir / f'{task}_{metric_name}.png')

                self.history.plot_ncm('train', task, 'F1', log_dir, task_info['num_classes'])
                self.history.plot_ncm('valid', task, 'F1', log_dir, task_info['num_classes'])

        avg_loss = self.trainer.logged_metrics['train_loss']
        self.history.update(phase="train", task="all", metric="avg_loss", value=avg_loss.item(), epoch=self.current_epoch)
        self.log('train_loss', avg_loss.item(), prog_bar=True, logger=True, on_epoch=True)
        self.history.plot('all', 'avg_loss', log_dir / 'loss.png')
        self.history.save(Path(self.trainer.log_dir).parents[1])

        self.train_preds = []
        self.train_targets = []
        self.train_masks = []

        self.shake_coeffs = torch.rand(len(self.loss_fns)) if self.shake else torch.ones(len(self.loss_fns))


    def validation_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']] # list of shapes [(B,T,F) or (B,L,T,F), ...]
        x_masks = [batch[f'{feature_name}_mask'] for feature_name in self.config['feature_list']] # list of shapes [(B,T), ...]
        x_mask = batch[f'{self.config["feature_list"][-1]}_mask']

        preds_heads = self(x, x_masks, names=self.config['feature_list']) # list of shapes [(B, T, n_classes_ec), (B, T, n_classes_ei), (B, T, 2)]  - emotion class, emotion intensity, valence & arousal

        y = [batch[task] for task in self.tasks] # list of shapes [(B,) or (B,T), ...]
        y_masks = [batch[f'{task}_mask'] for task in self.tasks] # list of shapes [(B, T), ...]

        loss = multitarget_loss(
            preds_heads, y, y_masks,
            self.loss_fns,
            self.shake_coeffs
        )

        self.log('valid_loss', loss, prog_bar=True, logger=True, on_epoch=True)
        self.valid_preds.append(preds_heads)
        self.valid_targets.append(y)
        self.valid_masks.append(y_masks)

        return loss


    def on_validation_epoch_end(self):
        preds = [torch.cat([batch[i] for batch in self.valid_preds], dim=0) for i in range(len(self.tasks))]
        targets = [torch.cat([batch[i] for batch in self.valid_targets], dim=0) for i in range(len(self.tasks))]
        masks = [torch.cat([batch[i] for batch in self.valid_masks], dim=0) for i in range(len(self.tasks))]

        for index, task in enumerate(self.tasks):
            task_info = self.config['tasks'][task]

            if task_info['type'] == 'classification':
                metrics = classification_metrics(y_true=targets[index], y_pred=preds[index], y_mask=masks[index], n_classes=task_info['num_classes'])
                for metric_name, metric_value in metrics.items():
                    self.history.update(phase="valid", task=task, metric=metric_name, value=metric_value, epoch=self.current_epoch)

                    if metric_name in task_info['metrics']:
                        self.log(f'valid_{task}_{metric_name}', metric_value, prog_bar=True, logger=True, on_epoch=True)

        #results_emotion_intensity = classification_metrics(preds[1], targets[1], masks[1], 2)
        #results_valence = va_metrics(preds[2], targets[2], masks[2])
        #results_arousal = va_metrics(preds[3], targets[3], masks[3])
        #self.log('valid_va_rmse', 0.5*(results_valence['rmse']+results_arousal['rmse']), prog_bar=True, logger=True)
        #self.log('valid_va_sagr', 0.5*(results_valence['sagr']+results_arousal['sagr']), prog_bar=True, logger=True)
        #self.log('valid_va_ccc', 0.5*(results_valence['ccc']+results_arousal['ccc']), prog_bar=True, logger=True)
        #self.log('valid_valence_rmse', results_valence['rmse'], prog_bar=True, logger=True)
        #self.log('valid_valence_sagr', results_valence['sagr'], prog_bar=True, logger=True)
        #self.log('valid_valence_ccc', results_valence['ccc'], prog_bar=True, logger=True)
        #self.log('valid_arousal_rmse', results_arousal['rmse'], prog_bar=True, logger=True)
        #self.log('valid_arousal_sagr', results_arousal['sagr'], prog_bar=True, logger=True)
        #self.log('valid_arousal_ccc', results_arousal['ccc'], prog_bar=True, logger=True)

        # Store for plotting
        avg_loss = self.trainer.logged_metrics['valid_loss']
        self.history.update(phase="valid", task="all", metric="avg_loss", value=avg_loss.item(), epoch=self.current_epoch)
        self.log('valid_loss', avg_loss.item(), prog_bar=True, logger=True, on_epoch=True)

        # valence
        #self.valid_valence_rmse.append(results_valence['rmse'])
        #self.valid_valence_sagr.append(results_valence['sagr'])
        #self.valid_valence_ccc.append(results_valence['ccc'])

        # arousal
        #self.valid_arousal_rmse.append(results_arousal['rmse'])
        #self.valid_arousal_sagr.append(results_arousal['sagr'])
        #self.valid_arousal_ccc.append(results_arousal['ccc'])

        #self.valid_va_rmse.append(0.5 * (results_valence['rmse'] + results_arousal['rmse']))
        #self.valid_va_sagr.append(0.5 * (results_valence['sagr'] + results_arousal['sagr']))
        #self.valid_va_ccc.append(0.5 * (results_valence['ccc'] + results_arousal['ccc']))

        #plot_dir = Path(self.trainer.log_dir).parents[1] if self.trainer.log_dir is not None else 'test'
        #plot_sentiment_metrics(self, plot_dir)

        # Clear the collected predictions and targets
        self.valid_preds = []
        self.valid_targets = []
        self.valid_masks = []


    def test_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']] # list of shapes [(B,T,F) or (B,L,T,F), ...]
        x_masks = [batch[f'{feature_name}_mask'] for feature_name in self.config['feature_list']] # list of shapes [(B,T), ...]
        x_mask = batch[f'{self.config["feature_list"][-1]}_mask']

        preds_heads = self(x, x_masks, names=self.config['feature_list']) # list of shapes [(B, T, n_classes_ec), (B, T, n_classes_ei), (B, T, 2)]  - emotion class, emotion intensity, valence & arousal

        y = [batch[task] for task in self.tasks] # list of shapes [(B,) or (B,T), ...]
        y_masks = [batch[f'{task}_mask'] for task in self.tasks] # list of shapes [(B, T), ...]

        self.test_preds.append(preds_heads)
        self.test_targets.append(y)
        self.test_masks.append(y_masks)


    def on_test_epoch_end(self):
        preds = [torch.cat([batch[i] for batch in self.test_preds], dim=0) for i in range(len(self.tasks))]
        targets = [torch.cat([batch[i] for batch in self.test_targets], dim=0) for i in range(len(self.tasks))]
        masks = [torch.cat([batch[i] for batch in self.test_masks], dim=0) for i in range(len(self.tasks))]

        log_dir = Path(self.trainer.log_dir).parents[1] / 'test'
        for index, task in enumerate(self.tasks):
            task_info = self.config['tasks'][task]

            if task_info['type'] == 'classification':
                metrics = classification_metrics(y_true=targets[index], y_pred=preds[index], y_mask=masks[index], n_classes=task_info['num_classes'])
                for metric_name, metric_value in metrics.items():
                    self.history.update(phase="test", task=task, metric=metric_name, value=metric_value, epoch=self.current_epoch)

                    if metric_name in task_info['metrics']:
                        self.log(f'test_{task}_{metric_name}', metric_value, prog_bar=True, logger=True, on_epoch=True)

                self.history.plot_ncm('test', task, 'F1', log_dir, task_info['num_classes'])

        # Save test results
        self.history.save_test(log_dir)

        # Clear the collected predictions and targets
        self.test_preds = []
        self.test_targets = []
        self.test_masks = []


    def configure_optimizers(self):
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam')
        base_lr = float(optimizer_config.get('base_lr', 1e-3))
        weight_decay = float(optimizer_config.get('weight_decay', 0))

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

        lr_scheduler_config = config.get('lr_scheduler', {})
        lr_scheduler_name = lr_scheduler_config.get('name', 'ReduceLROnPlateau')

        if lr_scheduler_name == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.config['n_epochs'])  # Adjust T_max based on epochs
        elif lr_scheduler_name == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=base_lr, total_steps=self.trainer.estimated_stepping_batches
            )
        else: # ReduceLROnPlateau
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(lr_scheduler_config.get('factor', 0.1)),
                patience=int(lr_scheduler_config.get('patience', 10)),
                min_lr=1e-8
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": lr_scheduler_config.get('monitor', 'valid_loss')
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


def train_model(config: dict):
    torch.set_float32_matmul_precision(config.get('float32_matmul_precision', 'medium')) # medium, high, highest
    experiment_name = config.get('experiment_name', datetime.now().strftime("%Y%m%d-%H%M%S"))
    output_dir = Path(config.get('output_dir', 'results')) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the DataModule
    data_module = EmotionDataModule(config=config)

    # Initialize the model
    if config['model_name'] == 'LinT':
        model = LinT(config=config)
    elif config['model_name'] == 'LinMulT':
        model = LinMulT(config=config)
    else:
        raise ValueError(f'Unsupported model name {config["model_name"]}')

    lightning_model = ModelWrapper(model, config=config)

    callbacks = []
    for checkpoint_config in config['checkpoints']:
        callback = ModelCheckpoint(
            dirpath=Path(output_dir) / 'checkpoint',
            filename=f"{checkpoint_config['name']}" + "-{epoch:02d}",
            monitor=checkpoint_config['monitor'],
            mode=checkpoint_config['mode'],
            save_top_k=1,
            verbose=True,
            save_weights_only=True
        )
        callbacks.append(callback)

    config_es = config.get('early_stopping', False)
    if config_es:
        early_stopping = EarlyStopping(
            monitor=config_es['monitor'], # 'valid_loss'
            patience=config_es['patience'], # 10
            mode=config_es['mode'], # 'min'
            verbose=True
        )
        callbacks.append(early_stopping)

    csv_logger = CSVLogger(save_dir=str(output_dir), name="csv_logs")

    # Define the trainer
    trainer = L.Trainer(
        accelerator=config.get('accelerator', 'gpu'),
        devices=config.get('devices', [0]),
        max_epochs=config.get('n_epochs', 100),
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=csv_logger,
        num_sanity_val_steps=0,
        #limit_train_batches=0.1,
        #limit_val_batches=0.1,
        #limit_test_batches=0.1,
        #fast_dev_run=True,
        gradient_clip_val=1.0
    )

    # Train the model
    trainer.fit(lightning_model, datamodule=data_module)

    # Test the model on the test set
    print("Evaluating on the test set...")

    best_model_path = callbacks[1].best_model_path # checkpoint_valid_emotion_class_F1
    print(f"Loading best model from: {best_model_path}")
    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=best_model_path)
    test_results = trainer.test(lightning_model, datamodule=data_module)


def parse_additional_args(unknown_args):
    additional_args = {}
    for i in range(0, len(unknown_args), 2):
        key = unknown_args[i].lstrip("--") # remove leading '--'
        value = unknown_args[i + 1]

        # Try to cast to integer or float if possible
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass # keep as string if it's not a number

        additional_args[key] = value

    if 'gpu_id' in additional_args:
        additional_args['devices'] = [additional_args.get('gpu_id', 0)]

    return additional_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script for EmotionLinMulT")
    parser.add_argument("--db_config_path", type=str, default='configs/unimodal/ravdess/emotion_class/opengraphau-fabnet-clip/dataloader.yaml', help="path to the dataset config file")
    parser.add_argument("--model_config_path", type=str, default='configs/unimodal/ravdess/emotion_class/opengraphau-fabnet-clip/model.yaml', help="path to the model config file")
    parser.add_argument("--train_config_path", type=str, default='configs/unimodal/ravdess/emotion_class/opengraphau-fabnet-clip/train.yaml', help="path to the train config file")
    args, unknown = parser.parse_known_args()

    config = {}
    config |= load_config(args.db_config_path)
    config |= load_config(args.model_config_path)
    config |= load_config(args.train_config_path)

    additional_args = parse_additional_args(unknown)
    config.update(additional_args)
    pprint(config)

    train_model(config)