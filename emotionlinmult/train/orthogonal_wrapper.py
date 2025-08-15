import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from emotionlinmult.train.history import History
from emotionlinmult.train.loss import multitarget_loss, AutomaticWeightedLoss
from emotionlinmult.train.metrics_running import RunningMetrics
from torch.nn.utils import clip_grad_norm_


class OrthogonalModelWrapper(L.LightningModule):

    def __init__(self, model, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters('config')
        self.automatic_optimization = False
        
        self.model = model

        #if config.get('lembda_augmentor', False): 
        #    self.augmentors = nn.ModuleList([
        #        FeatureAugmentor(768), # wavlm baseplus
        #        FeatureAugmentor(1024), # clip
        #    ])

        self.tasks = config['tasks']
        #self.consistency_rules = config.get('consistency_rules', [])
        self.stage = config.get('stage', 'tmm_pretraining')  # 'tmm_pretraining' for pretraining, 'finetuning' for finetuning

        # Initialize metrics for each phase
        self.train_metrics = RunningMetrics(self.tasks)
        self.valid_metrics = RunningMetrics(self.tasks)
        self.test_metrics = RunningMetrics(self.tasks)

        # Store predictions and targets for test set (for final evaluation)
        self.test_benchmark = []

        self.log_dir = Path(config['experiment_dir'])
        self.history = History(self.log_dir)

        if self.stage not in {'tmm_pretraining', 'finetuning'}:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'tmm_pretraining' or 'finetuning'.")
        
        if 'awl' in self.config:
            if self.stage == 'finetuning':
                num_task = len([task for task in self.tasks if 'tmm_' not in task])
            else: # pretraining
                num_task = len([task for task in self.tasks if 'tmm_' in task])
            self.awl = AutomaticWeightedLoss(num_task)
        else:
            self.awl = None


    @staticmethod
    def gram_schmidt_project(grads_by_task):
        # grads_by_task: list of [tensor] per task
        # Gram-Schmidt: project each task's gradients onto subspace orthogonal to all previous tasks
        projected = []
        for t, g_t in enumerate(grads_by_task):
            g_proj = []
            for i, g in enumerate(g_t):
                if g is None:
                    g_proj.append(None)
                    continue
                # Subtract previous tasks' components
                gs_prev = [projected[s][i] for s in range(len(projected)) if projected[s][i] is not None]
                # Project away previous tasks' directions
                for g_prev in gs_prev:
                    alpha = torch.dot(g.flatten(), g_prev.flatten())
                    norm = torch.dot(g_prev.flatten(), g_prev.flatten())
                    if norm > 0:
                        g = g - (alpha/norm) * g_prev
                g_proj.append(g)
            projected.append(g_proj)
        return projected


    def aux_loss_scheduling(self, current_epoch):
        if current_epoch < 10:
            # First phase: 0.6 main, 0.2 aux1, 0.2 aux2
            main_weight = 0.8
            aux_weight = 0.1
            num_modalities = len(self.config['feature_list'])
            modality_wise = [aux_weight / num_modalities for _ in range(num_modalities)]
            return main_weight, modality_wise
        #elif current_epoch < 15:
        #    # Second phase: 0.8 main, 0.1 aux1, 0.1 aux2
        #    main_weight = 0.8
        #    aux_weight = 0.2
        #    num_modalities = len(self.config['feature_list'])
        #    modality_wise = [aux_weight / num_modalities for _ in range(num_modalities)]
        #    return main_weight, modality_wise
        else:
            # Final phase: 1.0 main, 0 aux1, 0 aux2
            return 1.0, [0.0, 0.0]


    def forward(self, x, masks=None):
        return self.model(x, masks)


    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        # Get batches
        #x = [batch[feature_name] for feature_name in self.config['feature_list']]  # list of shapes [(B, T, F), ...]
        #x_masks = [batch[f'{feature_name}_mask'] for feature_name in self.config['feature_list']]  # list of shapes [(B, T), ...]
        #y_true = [batch[task] for task in self.tasks]  # list of shapes [(B,) or (B, T), ...]
        #y_true_masks = [batch[f'{task}_mask'] for task in self.tasks]  # list of shapes [(B,) or (B, T), ...]

        #head_params = [params for params_name, params in self.model.named_parameters() if params_name.startswith('output_heads')]
        shared_params = [params for params_name, params in self.model.named_parameters() if not params_name.startswith('output_heads')]
        head_params_by_task = {}
        for task_name in self.tasks:
            head_params_by_task[task_name] = [
                p for pname, p in self.model.named_parameters()
                if pname.startswith('output_heads') and task_name in pname.split('.')
            ]
        shared_grads, head_grads = {}, {}

        total_loss = 0.0
        task_losses = {}

        # Filter batches to tasks
        # Compute loss and gradient per task
        for task_ind, (task_name, task_info) in enumerate(self.tasks.items()):
            
            # video_level
            if task_name in {'emotion_class', 'emotion_intensity', 'sentiment'}:
                mask_b = batch[f'{task_name}_mask'] # sample-level usability mask with shape (B,)
                if sum(mask_b) == 0:
                    task_losses[task_name] = np.nan # this should be np.nan or sth which can be used later on as a missing value in the plots
                    continue # no sample in the batch for a given task, skip
                task_x = [batch[feature_name][mask_b] for feature_name in self.config['feature_list']]
                task_x_masks = [batch[f'{feature_name}_mask'][mask_b] for feature_name in self.config['feature_list']]
                task_y_true = [batch[task_name][mask_b]]
                task_y_true_masks = [batch[f'{task_name}_mask'][mask_b]]
            else: # sequence_level {'emotion_class_fw', 'valence', 'arousal'}
                # find samples which are useable for this task
                mask_b  = batch[f'{task_name}_mask'].any(dim=1) # sample-level usability mask with shape (B,)
                if sum(mask_b) == 0:
                    task_losses[task_name] = np.nan # this should be np.nan or sth which can be used later on as a missing value in the plots
                    continue # no sample in the batch for a given task, skip
                task_x = [batch[feature_name][mask_b] for feature_name in self.config['feature_list']]
                task_x_masks = [batch[f'{feature_name}_mask'][mask_b] for feature_name in self.config['feature_list']]
                task_y_true = [batch[task_name][mask_b]]
                task_y_true_masks = [batch[f'{task_name}_mask'][mask_b]]
            if self.config.get('auxiliary_heads', False):
                task_preds_heads, task_all_aux_heads = self(task_x, task_x_masks)
            else:
                task_preds_heads = self(task_x, task_x_masks)
            task_active_preds_heads = {task_name: task_preds_heads[task_name]}
            # Compute main loss for this task
            loss = multitarget_loss(task_active_preds_heads, task_y_true, task_y_true_masks, {task_name: task_info}, awl=self.awl)
            total_loss += loss
            task_losses[task_name] = loss.item()
            self.manual_backward(loss, retain_graph=True if task_ind < len(self.tasks) - 1 else False) # last graph is not retained
            shared_grads[task_name] = [p.grad.clone() if p.grad is not None else None for p in shared_params]
            head_grads[task_name] = [p.grad.clone() if p.grad is not None else None for p in head_params_by_task[task_name]]

            self.train_metrics.update(
                task_name=task_name,
                preds=task_active_preds_heads[task_name].detach().cpu(),
                target=task_y_true[0].detach().cpu(),
                mask=task_y_true_masks[0].detach().cpu()
            )

            optimizer.zero_grad()

        # Orthogonalize gradients (Gram-Schmidt)
        projected_grads = self.gram_schmidt_project([shared_grads[tn] for tn in self.tasks]) #grads_by_task)

        # set shared grads (orthogonalized)
        for i, p in enumerate(shared_params):
            grads_to_sum = [g[i] for g in projected_grads if g[i] is not None]
            p.grad = sum(grads_to_sum) if grads_to_sum else None

        # set each head's grads (NO orthogonalization)
        for task_name in self.tasks:
            for p, g in zip(head_params_by_task[task_name], head_grads[task_name]):
                p.grad = g.clone() if g is not None else None

        # Clip gradients if defined
        max_norm = self.config.get('clip_grad_norm', None) # 1.0 is in config file
        if max_norm is not None:
            clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], max_norm=max_norm)


        optimizer.step()
        optimizer.zero_grad()

        # Log metrics
        avg_loss = total_loss / len(self.tasks)
        self.log("train_loss", avg_loss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=False, batch_size=self.config['batch_size'])
        for task_name, value in task_losses.items():
            self.log(f"train_loss/{task_name}", value, prog_bar=False, logger=True,
                     on_step=True, on_epoch=False, batch_size=self.config['batch_size'])
        return avg_loss


    def on_train_epoch_end(self):
        # training_step -> validation_step -> on_valid_epoch_end -> on_train_epoch_end
        # Log metrics for training
        train_metrics = self.train_metrics.compute()
        
        values = []
        for task_name, metric_dict in train_metrics.items():
            if task_name == 'emotion_class':
                values.append(metric_dict['F1'].item())
            elif task_name == 'emotion_class_fw':
                values.append(metric_dict['F1'].item())
            elif task_name == 'emotion_intensity':
                values.append(metric_dict['F1'].item())
            elif task_name == 'sentiment':
                values.append(metric_dict['F1_7'].item())
            elif task_name == 'valence':
                values.append(metric_dict['CCC'].item())
            elif task_name == 'arousal':
                values.append(metric_dict['CCC'].item())
        composite_score = float(np.mean(values))

        self.log("train_combined_score", composite_score, prog_bar=True, logger=True, on_epoch=True)
        self.history.update(
            phase="train", 
            task="all", 
            metric="combined_score", 
            value=composite_score, 
            epoch=self.current_epoch
        )
        self.history.plot("all", "combined_score")

        for task_name, metrics in train_metrics.items():
            task_info = self.tasks[task_name]
            for metric_name, metric_value in metrics.items():
                # Update history for plotting
                self.history.update(
                    phase="train", 
                    task=task_name, 
                    metric=metric_name, 
                    value=metric_value.item(), 
                    epoch=self.current_epoch
                )
                
                # Log to logger if in the task's metrics list
                if metric_name in task_info.get('metrics', []):
                    self.log(
                        f'train_{task_name}_{metric_name}', 
                        metric_value,
                        prog_bar=True,
                        logger=True,
                        on_epoch=True,
                        batch_size=self.config['batch_size']
                    )
                    
                    # Plot metrics
                    self.history.plot(task_name, metric_name)

        # Log average training loss
        if 'train_loss' in self.trainer.logged_metrics:
            avg_loss = self.trainer.logged_metrics['train_loss']
            self.history.update(
                phase="train", 
                task="all", 
                metric="avg_loss", 
                value=avg_loss.item(), 
                epoch=self.current_epoch
            )
            self.log('train_loss', avg_loss, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config['batch_size'])
        
        for task_name, task_info in self.tasks.items():
            if f'train_loss/{task_name}' in self.trainer.logged_metrics:
                task_loss = self.trainer.logged_metrics[f'train_loss/{task_name}']
                self.history.update(
                    phase="train", 
                    task=task_name, 
                    metric=f'train_loss/{task_name}',
                    value=task_loss.item(), 
                    epoch=self.current_epoch
                )
                self.log(f'train_loss/{task_name}', task_loss, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config['batch_size'])
                self.history.plot(task_name, f'train_loss/{task_name}')

        self.history.plot('all', 'avg_loss')
        
        # Save history and record epoch time
        self.history.record_epoch_time()
        self.history.save()

        if self.awl is not None:
            print('awl weights:', [round(float(elem.detach().cpu()), 2) for elem in self.awl.params])


    def validation_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']]
        x_masks = [batch[f'{feature_name}_mask'] for feature_name in self.config['feature_list']]
        y_true = [batch[task] for task in self.tasks]
        y_true_masks = [batch[f'{task}_mask'] for task in self.tasks]

        # Forward pass
        if self.config.get('auxiliary_heads', False):
            preds_heads, all_aux_heads = self(x, x_masks)
        else:
            preds_heads = self(x, x_masks)
        
        active_preds_heads = {task_name: preds_heads[task_name] for task_name in self.tasks}

        # Compute main loss
        loss = multitarget_loss(active_preds_heads, y_true, y_true_masks, self.tasks, awl=self.awl)

        # Calculate auxiliary losses
        #current_epoch = self.trainer.current_epoch
        #if self.config.get('auxiliary_heads', False) and current_epoch < 10:
        #    main_weight, aux_weights = self.aux_loss_scheduling(current_epoch)
        #    loss = main_weight * loss
        #    assert len(all_aux_heads) == len(aux_weights)
        #    ignore_sequence_tasks = {'emotion_class_fw', 'valence', 'arousal'}
        #    for ind, (aux_heads, aux_weight) in enumerate(zip(all_aux_heads, aux_weights)):
        #        active_aux_heads = {task: aux_heads[task + f'_{ind}'] for task in self.tasks if task not in ignore_sequence_tasks}
        #        y_true_aux = [batch[task] for task in self.tasks if task not in ignore_sequence_tasks]
        #        y_true_masks_aux = [batch[f'{task}_mask'] for task in self.tasks if task not in ignore_sequence_tasks]
        #        tasks_aux = {task: self.tasks[task] for task in self.tasks if task not in ignore_sequence_tasks}
        #        aux_loss = multitarget_loss(active_aux_heads, y_true_aux, y_true_masks_aux, tasks_aux, awl=None)
        #        loss += aux_weight * aux_loss

        # Update metrics
        for i, (task_name, task_info) in enumerate(self.tasks.items()):
            mask = y_true_masks[i].detach().cpu()
            
            # video_level
            if task_name in {'emotion_class', 'emotion_intensity', 'sentiment'}:
                if mask.sum() == 0: continue
            else: # sequence_level {'emotion_class_fw', 'valence', 'arousal'}
                if mask.any(dim=1).sum() == 0: continue

            pred = active_preds_heads[task_name].detach().cpu()
            target = y_true[i].detach().cpu()
            self.valid_metrics.update(task_name, pred, target, mask)

        # Log validation loss
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config['batch_size'])

        return loss


    def on_validation_epoch_end(self):
        # Compute metrics for validation
        valid_metrics = self.valid_metrics.compute()

        values = []
        for task_name, metric_dict in valid_metrics.items():
            if task_name == 'emotion_class':
                values.append(metric_dict['F1'].item())
            elif task_name == 'emotion_class_fw':
                values.append(metric_dict['F1'].item())
            elif task_name == 'emotion_intensity':
                values.append(metric_dict['F1'].item())
            elif task_name == 'sentiment':
                values.append(metric_dict['F1_7'].item())
            elif task_name == 'valence':
                values.append(metric_dict['CCC'].item())
            elif task_name == 'arousal':
                values.append(metric_dict['CCC'].item())
        composite_score = float(np.mean(values))

        self.log("valid_combined_score", composite_score, prog_bar=True, logger=True, on_epoch=True)
        self.history.update(
            phase="valid", 
            task="all", 
            metric="combined_score", 
            value=composite_score, 
            epoch=self.current_epoch
        )
        # self.history.plot("all", "combined_score")

        for task_name, metrics in valid_metrics.items():
            task_info = self.tasks[task_name]
            for metric_name, metric_value in metrics.items():
                # Update history for plotting
                self.history.update(
                    phase="valid", 
                    task=task_name, 
                    metric=metric_name, 
                    value=metric_value.item(), 
                    epoch=self.current_epoch
                )

                # Log to logger if in the task's metrics list
                if metric_name in task_info.get('metrics', []):
                    self.log(
                        f'valid_{task_name}_{metric_name}', 
                        metric_value,
                        prog_bar=True,
                        logger=True,
                        on_epoch=True,
                        batch_size=self.config['batch_size']
                    )

                    # Plot metrics
                    self.history.plot(task_name, metric_name)

        # Log average validation loss
        if 'valid_loss' in self.trainer.logged_metrics:
            avg_loss = self.trainer.logged_metrics['valid_loss']
            self.history.update(
                phase="valid", 
                task="all", 
                metric="avg_loss", 
                value=avg_loss.item(), 
                epoch=self.current_epoch
            )

        # Save history
        self.history.save()


    def test_step(self, batch, batch_idx):
        # Skip test step in pretraining phase
        if self.stage == 'tmm_pretraining':
            return None

        # Prepare inputs and targets
        x = [batch[feature_name] for feature_name in self.config['feature_list']]  # # list of shapes [(B, T, F), ...]
        x_masks = [batch[f'{feature_name}_mask'] for feature_name in self.config['feature_list']]  # # list of shapes [(B, T), ...]
        y_true = [batch[task] for task in self.tasks]  # list of shapes [(B, T, F), ...] or [(B, F), ...]
        y_true_masks = [batch[f'{task}_mask'] for task in self.tasks]  # list of shapes [(B, T), ...] or [(B, F), ...]

        # Forward pass
        if self.config.get('auxiliary_heads', False):
            preds_heads, all_aux_heads = self(x, x_masks)
        else:
            preds_heads = self(x, x_masks)

        active_preds_heads = {task_name: preds_heads[task_name] for task_name in self.tasks}

        # Update metrics and store predictions for each task
        for i, (task_name, task_info) in enumerate(self.tasks.items()):
            pred = active_preds_heads[task_name].detach().cpu()
            target = y_true[i].detach().cpu()
            mask = y_true_masks[i].detach().cpu()
            self.test_metrics.update(task_name, pred, target, mask)

            # skip tmm task predictions
            if task_name in ['tmm_wavlm_baseplus', 'tmm_clip', 'tmm_xml_roberta']: continue

            # Store predictions and targets
            batch_info = {
                'pred': pred,
                'target': target,
                'mask': mask,
                'task': task_name,
                'dataset': batch['datasets'],
                'key': batch['keys']
            }
            self.test_benchmark.append(batch_info)


    def on_test_epoch_end(self):
        # Skip test epoch end in pretraining phase
        if self.stage == 'tmm_pretraining':
            return

        # Compute test metrics
        test_metrics = self.test_metrics.compute()
        
        for task_name, metrics in test_metrics.items():
            task_info = self.tasks[task_name]
            for metric_name, metric_value in metrics.items():
                # Update history for plotting
                self.history.update(
                    phase="test",
                    task=task_name,
                    metric=metric_name,
                    value=metric_value.item(),
                    epoch=self.current_epoch
                )
                
                # Log to logger if this is a tracked metric for this task
                if metric_name in task_info.get('metrics', []):
                    self.log(
                        f'test_{task_name}_{metric_name}',
                        metric_value,
                        prog_bar=True,
                        logger=True,
                        on_epoch=True,
                        batch_size=self.config['batch_size']
                    )

        # Save test history
        self.history.save_test()
        
        # Process and save predictions
        predictions = self._process_predictions(self.test_benchmark)
        self.save_predictions(predictions)
        
        # Clean up
        self.test_benchmark = []


    def _process_predictions(self, test_benchmark):
        """Process test predictions and return organized predictions by dataset and task.
        
        Args:
            test_benchmark: List of dictionaries containing prediction data
            
        Returns:
            Dictionary containing organized predictions by dataset and task
        """
        task_batches = {}
        for batch_info in test_benchmark:
            task_name = batch_info['task']
            if task_name not in task_batches:
                task_batches[task_name] = []
            task_batches[task_name].append(batch_info)

        dataset_predictions = {}
        for task_name, batch_info_list in task_batches.items():
            if task_name not in self.tasks: continue

            for batch_info in batch_info_list:
                for i in range(len(batch_info['key'])):
                    dataset = batch_info['dataset'][i]
                    sample_key = batch_info['key'][i]

                    if dataset not in dataset_predictions:
                        dataset_predictions[dataset] = {}
                    if task_name not in dataset_predictions[dataset]:
                        dataset_predictions[dataset][task_name] = {}

                    pred_value, target_value, mask_value = self.convert_to_appropriate_format(task_name, batch_info['pred'][i], batch_info['target'][i], batch_info['mask'][i])

                    dataset_predictions[dataset][task_name][sample_key] = {
                        'y_pred': pred_value,
                        'y_true': target_value,
                        'mask': mask_value
                    }

        return dataset_predictions
    

    def convert_to_appropriate_format(self, task_name, pred, target, mask):

        if task_name == 'sentiment':
            # Scalar regression () -> float
            pred_value = float(pred.item())
            target_value = float(target.item())
            mask_value = bool(mask.item())
            
        elif task_name in ['sentiment_class', 'emotion_class', 'emotion_intensity']:
            # Classification (C,) -> list of logits and class index
            pred_value = [float(x) for x in pred.tolist()]
            target_value = int(target.item())
            mask_value = bool(mask.item())
            
        elif task_name in ['valence', 'arousal']:
            # Sequence regression (T,) -> dict of frame_id: float
            pred_frames = pred.squeeze(-1)
            target_frames = target.squeeze(-1)
            
            pred_value = {}
            target_value = {}
            mask_value = {}

            for frame_id in range(len(pred_frames)):
                pred_value[str(frame_id)] = float(pred_frames[frame_id].item())
                target_value[str(frame_id)] = float(target_frames[frame_id].item())
                mask_value[str(frame_id)] = bool(mask[frame_id].item())
            
        elif task_name == 'emotion_class_fw':
            # Sequence classification (T, C) -> dict of frame_id: list of logits and class index
            pred_value = {}
            target_value = {}
            mask_value = {}
            
            for frame_id in range(len(pred)):
                pred_value[str(frame_id)] = [float(x) for x in pred[frame_id].tolist()]
                target_value[str(frame_id)] = int(target[frame_id].item())
                mask_value[str(frame_id)] = bool(mask[frame_id].item())
        
        else:
            raise ValueError(f'Unknown task: {task_name}')
        
        return pred_value, target_value, mask_value


    def save_predictions(self, predictions):
        """Save predictions to JSON files organized by dataset.
        
        Args:
            predictions: Dictionary containing organized predictions by dataset and task
        """
        if 'model_stage2_cp_path' not in self.config:
            output_dir = Path(self.config['experiment_dir']) / 'predictions' / 'auto'
        else:
            output_dir = Path(self.config['experiment_dir']) / 'predictions' / Path(self.config['model_stage2_cp_path']).stem
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for dataset, tasks in predictions.items():
            output_path = output_dir / f'test_{dataset.lower()}.json'

            if output_path.exists():
                print(str(output_path) + ' already exists... Skip.')
                continue
            else:
                print('Saving predictions to ' + str(output_path))

            with open(output_path, 'w') as f:
                json.dump(tasks, f, indent=2)
            
            print(str(output_path) + ' saved.')

        print(f"Predictions saved to {output_dir}")


    def configure_optimizers(self):
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam')
        base_lr = float(optimizer_config.get('base_lr', 1e-3))
        weight_decay = float(optimizer_config.get('weight_decay', 0))

        # Configure optimizer
        if self.config.get('awl', False):
            parameters = [
                {'params': self.model.parameters()},
                {'params': self.awl.parameters(), 'lr': self.config.get('awl', {}).get('lr', 0.01)}
            ]
            if self.config.get('lembda_augmentor', False):
                parameters += [
                    {'params': self.augmentors.parameters()}
                ]
        else:
            parameters = self.model.parameters()
            if self.config.get('lembda_augmentor', False):
                parameters += self.augmentors.parameters()

        if optimizer_name == 'radam':
            optimizer = torch.optim.RAdam(
                parameters,
                lr=base_lr,
                weight_decay=weight_decay,
                decoupled_weight_decay=optimizer_config.get('decoupled_weight_decay', False)
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                parameters,
                lr=base_lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adams':
            from emotionlinmult.train.optim import AdamS
            optimizer = AdamS(
                parameters,
                lr=base_lr,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(parameters, lr=base_lr)

        # Configure learning rate scheduler
        lr_scheduler_config = self.config.get('lr_scheduler', {})
        lr_scheduler_name = lr_scheduler_config.get('name', 'ReduceLROnPlateau')
        

        # Configure warmup scheduler
        if self.config.get('warmup_epochs', 0) > 0:
            # write a warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config['warmup_epochs']
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
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['n_epochs']-self.config.get('warmup_epochs', 0))
        elif lr_scheduler_name == 'CosineAnnealingWarmRestarts':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-7
            )
        else:
            raise ValueError(f'Given lr scheduler is not supported: {lr_scheduler_name}')

        # Combine warmup and main scheduler using SequentialLR
        if self.config.get('warmup_epochs', 0) > 0:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.config['warmup_epochs']]
            )
        else:
            lr_scheduler = main_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

