from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score,
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
)
from torchmetrics.regression import (
    MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, 
    ConcordanceCorrCoef
)

class SentimentMetrics:
    """Metrics for sentiment analysis (MOSEI) with regression [-3, 3]"""
    def __init__(self):
        # Metrics will be initialized on first update with the correct device
        self.mae = None
        self.corr = None
        self.acc2 = None
        self.precision2 = None
        self.recall2 = None
        self.f1_2 = None
        self.acc7 = None
        self.precision7 = None
        self.recall7 = None
        self.f1_7 = None
        self._initialized = False
        
    def _initialize_metrics(self, device):
        """Initialize metrics on the correct device"""
        # Regression metrics
        self.mae = MeanAbsoluteError().to(device)
        self.corr = PearsonCorrCoef().to(device)
        
        # Classification metrics for binary (positive/negative)
        self.acc2 = BinaryAccuracy(threshold=0.0).to(device)
        self.precision2 = BinaryPrecision(threshold=0.0).to(device)
        self.recall2 = BinaryRecall(threshold=0.0).to(device)
        self.f1_2 = BinaryF1Score(threshold=0.0).to(device)
        
        # Classification metrics for 7-class
        self.acc7 = MulticlassAccuracy(num_classes=7, average='weighted').to(device)
        self.precision7 = MulticlassPrecision(num_classes=7, average='weighted').to(device)
        self.recall7 = MulticlassRecall(num_classes=7, average='weighted').to(device)
        self.f1_7 = MulticlassF1Score(num_classes=7, average='weighted').to(device)
        
        self._initialized = True
        
    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None):
        # Initialize metrics on first update if needed
        if not hasattr(self, '_initialized') or not self._initialized:
            self._initialize_metrics(device=preds.device)

        # Move target to same device as preds if needed
        target = target.to(preds.device)

        if mask is not None:
            mask = mask.to(preds.device)
            preds = preds[mask]
            target = target[mask]

        if target.numel() == 0:  # Skip if no valid targets
            return

        if preds.ndim > 1 and preds.shape[-1] == 1: # (B, 1)
            preds = preds.squeeze(-1)
        if target.ndim > 1 and target.shape[-1] == 1: # (B, 1)
            target = target.squeeze(-1)

        # Regression metrics
        self.mae.update(preds, target)
        self.corr.update(preds, target)

        # Convert to class predictions
        # Binary: negative (<=0) vs positive (>0)
        binary_target = (target > 0).long()
        binary_pred = (preds > 0).long()

        # 7-class: map [-3,3] to [0,6]
        target_classes = torch.clamp((target + 3).round(), 0, 6).long()
        pred_classes = torch.clamp((preds + 3).round(), 0, 6).long()

        # Update binary metrics
        self.acc2.update(binary_pred, binary_target)
        self.precision2.update(binary_pred, binary_target)
        self.recall2.update(binary_pred, binary_target)
        self.f1_2.update(binary_pred, binary_target)
        
        # Update 7-class metrics
        self.acc7.update(pred_classes, target_classes)
        self.precision7.update(pred_classes, target_classes)
        self.recall7.update(pred_classes, target_classes)
        self.f1_7.update(pred_classes, target_classes)
    
    def compute(self) -> Dict[str, Tensor]:
        if not hasattr(self, '_initialized') or not self._initialized:
            # Return default values if no updates happened
            return {
                'ACC_2': torch.tensor(0.0),
                'F1_2': torch.tensor(0.0),
                'ACC_7': torch.tensor(0.0),
                'F1_7': torch.tensor(0.0),
                'MAE': torch.tensor(0.0),
                'CORR': torch.tensor(0.0),
            }
            
        results = {
            'ACC_2': self.acc2.compute() if hasattr(self, 'acc2') and self.acc2 is not None else torch.tensor(0.0),
            'F1_2': self.f1_2.compute() if hasattr(self, 'f1_2') and self.f1_2 is not None else torch.tensor(0.0),
            'ACC_7': self.acc7.compute() if hasattr(self, 'acc7') and self.acc7 is not None else torch.tensor(0.0),
            'F1_7': self.f1_7.compute() if hasattr(self, 'f1_7') and self.f1_7 is not None else torch.tensor(0.0),
            'MAE': self.mae.compute() if hasattr(self, 'mae') and self.mae is not None else torch.tensor(0.0),
            'CORR': self.corr.compute() if hasattr(self, 'corr') and self.corr is not None else torch.tensor(0.0),
        }
        self.reset()
        return results
    
    def reset(self):
        if hasattr(self, 'mae') and self.mae is not None:
            self.mae.reset()
        if hasattr(self, 'corr') and self.corr is not None:
            self.corr.reset()
        if hasattr(self, 'acc2') and self.acc2 is not None:
            self.acc2.reset()
        if hasattr(self, 'precision2') and self.precision2 is not None:
            self.precision2.reset()
        if hasattr(self, 'recall2') and self.recall2 is not None:
            self.recall2.reset()
        if hasattr(self, 'f1_2') and self.f1_2 is not None:
            self.f1_2.reset()
        if hasattr(self, 'acc7') and self.acc7 is not None:
            self.acc7.reset()
        if hasattr(self, 'precision7') and self.precision7 is not None:
            self.precision7.reset()
        if hasattr(self, 'recall7') and self.recall7 is not None:
            self.recall7.reset()
        if hasattr(self, 'f1_7') and self.f1_7 is not None:
            self.f1_7.reset()
        torch.cuda.empty_cache()

class ValenceArousalMetrics:
    """Metrics for valence/arousal prediction [-1, 1]"""
    def __init__(self):
        # Initialize metrics
        self.ccc = None
        self.rmse = None
        self.pcc = None
        self.sagr = None
        self._initialized = False
        
    def _initialize_metrics(self, device):
        """Initialize metrics on the correct device"""
        self.ccc = ConcordanceCorrCoef().to(device)
        self.rmse = MeanSquaredError(squared=False).to(device)  # RMSE instead of MSE
        self.pcc = PearsonCorrCoef(num_outputs=1).to(device)
        self.sagr = MeanMetric().to(device)  # Sign Agreement Ratio
        self._initialized = True
        
    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None):
        """Update metrics with predictions and targets.
        
        Args:
            preds: Predictions tensor of shape (B, 1) or (B, T, 1)
            target: Target tensor of same shape as preds
            mask: Optional boolean mask of same shape as preds
        """
        # Skip if no valid samples
        if mask is not None and not mask.any():
            return
            
        # Initialize metrics on first update if needed
        if not hasattr(self, '_initialized') or not self._initialized:
            self._initialize_metrics(device=preds.device)
            
        # Move target to same device as preds if needed
        target = target.to(preds.device)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.to(preds.device)
            preds = preds[mask]
            target = target[mask]
        
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.squeeze(-1)
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)
        
        # Flatten time dimension if present
        if preds.dim() > 2:  # (B, T, 1)
            preds = preds.flatten()  # (B*T,)
            target = target.flatten()
        
        # Skip if no valid predictions after masking
        if len(preds) == 0:
            return
            
        self.ccc.update(preds, target)
        self.rmse.update(preds, target)
        self.pcc.update(preds, target)
            
        # Sign agreement
        same_sign = (preds * target) > 0
        self.sagr.update(same_sign.float().mean())
    
    def compute(self) -> Dict[str, Tensor]:
        if not hasattr(self, '_initialized') or not self._initialized:
            # Return default values if no updates have been made
            return {
                'CCC': torch.tensor(0.0),
                'RMSE': torch.tensor(0.0),
                'PCC': torch.tensor(0.0),
                'SAGR': torch.tensor(0.0)
            }
            
        results = {
            'CCC': self.ccc.compute() if hasattr(self, 'ccc') and self.ccc is not None else torch.tensor(0.0),
            'RMSE': self.rmse.compute() if hasattr(self, 'rmse') and self.rmse is not None else torch.tensor(0.0),
            'PCC': self.pcc.compute() if hasattr(self, 'pcc') and self.pcc is not None else torch.tensor(0.0),
            'SAGR': self.sagr.compute() if hasattr(self, 'sagr') and self.sagr is not None else torch.tensor(0.0)
        }
        self.reset()
        return results
    
    def reset(self):
        if hasattr(self, 'ccc') and self.ccc is not None:
            self.ccc.reset()
        if hasattr(self, 'rmse') and self.rmse is not None:
            self.rmse.reset()
        if hasattr(self, 'pcc') and self.pcc is not None:
            self.pcc.reset()
        if hasattr(self, 'sagr') and self.sagr is not None:
            self.sagr.reset()
        torch.cuda.empty_cache()

class ClassificationMetrics:
    """Metrics for classification tasks (B, C) or (B, T, C)"""
    def __init__(self, num_classes: int, average: str = 'weighted', 
                 ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        self._initialized = False
        
        # Will be initialized on first update
        self.acc = None
        self.precision = None
        self.recall = None
        self.f1 = None
        
    def _initialize_metrics(self, device):
        """Initialize metrics on the correct device"""
        self.acc = MulticlassAccuracy(
            num_classes=self.num_classes, 
            average='weighted',
            ignore_index=self.ignore_index
        ).to(device)
        self.precision = MulticlassPrecision(
            num_classes=self.num_classes, 
            average=self.average,
            ignore_index=self.ignore_index
        ).to(device)
        self.recall = MulticlassRecall(
            num_classes=self.num_classes,
            average=self.average,
            ignore_index=self.ignore_index
        ).to(device)
        self.f1 = MulticlassF1Score(
            num_classes=self.num_classes,
            average=self.average,
            ignore_index=self.ignore_index
        ).to(device)
        self._initialized = True
    
    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None):
        """
        Args:
            preds: (B, C) or (B, T, C) tensor of logits
            target: (B,) or (B, T) tensor of target class indices
            mask: Optional boolean mask for valid positions
        """
        # Initialize metrics on first update if needed
        if not hasattr(self, '_initialized') or not self._initialized:
            self._initialize_metrics(device=preds.device)
            
        # Move target to same device as preds if needed
        target = target.to(preds.device)
        
        # Handle frame-wise classification (B, T, C) -> (B*T, C)
        if preds.dim() == 3:
            B, T, C = preds.shape
            preds = preds.reshape(-1, C)  # (B*T, C)
            target = target.reshape(-1)    # (B*T,)
            if mask is not None:
                mask = mask.reshape(-1)    # (B*T,)
        
        if mask is not None:
            mask = mask.to(preds.device)
            preds = preds[mask]
            target = target[mask]
            
        if target.numel() == 0:  # Skip if no valid targets
            return

        # Get predicted class indices if needed
        if preds.dim() > 1 and preds.size(-1) > 1:
            preds = preds.argmax(dim=-1)
                
        self.acc(preds, target)
        self.precision(preds, target)
        self.recall(preds, target)
        self.f1(preds, target)
    
    def compute(self) -> Dict[str, Tensor]:
        if not hasattr(self, '_initialized') or not self._initialized:
            # Return default values if no updates happened
            return {
                'ACC': torch.tensor(0.0),
                'P': torch.tensor(0.0),
                'R': torch.tensor(0.0),
                'F1': torch.tensor(0.0)
            }
            
        results = {
            'ACC': self.acc.compute() if hasattr(self, 'acc') and self.acc is not None else torch.tensor(0.0),
            'P': self.precision.compute() if hasattr(self, 'precision') and self.precision is not None else torch.tensor(0.0),
            'R': self.recall.compute() if hasattr(self, 'recall') and self.recall is not None else torch.tensor(0.0),
            'F1': self.f1.compute() if hasattr(self, 'f1') and self.f1 is not None else torch.tensor(0.0)
        }
        self.reset()
        return results
    
    def reset(self):
        if hasattr(self, 'acc') and self.acc is not None:
            self.acc.reset()
        if hasattr(self, 'precision') and self.precision is not None:
            self.precision.reset()
        if hasattr(self, 'recall') and self.recall is not None:
            self.recall.reset()
        if hasattr(self, 'f1') and self.f1 is not None:
            self.f1.reset()
        torch.cuda.empty_cache()


class ReconstructionMetrics:
    """Metrics for masked reconstruction tasks (tmm_clip, tmm_wavlm_baseplus, tmm_xml_roberta)"""
    def __init__(self):
        self.mse = None
        self.mae = None
        self.pcc = None
        self.num_samples = 0
        self.total_masked = 0
        self.seq_len = 0
        self._initialized = False
        
    def _initialize_metrics(self, device):
        """Initialize metrics on the correct device"""
        self.mse = MeanSquaredError().to(device)
        self.mae = MeanAbsoluteError().to(device)
        self.pcc = PearsonCorrCoef().to(device)
        self._initialized = True
        
    def update(self, preds: Tensor, target: Tensor, mask: Tensor):
        """
        Args:
            preds: (B, T, F) - Reconstructed features
            target: (B, T, F) - Original features
            mask: (B, T) - Boolean mask where True indicates masked positions
        """
        # Initialize metrics on first update if needed
        if not hasattr(self, '_initialized') or not self._initialized:
            self._initialize_metrics(device=preds.device)
        
        # Move tensors to correct device if needed
        target = target.to(preds.device)
        mask = mask.to(preds.device)
        
        # Flatten spatial dimensions
        B, T, F = preds.shape
        if not hasattr(self, 'seq_len') or self.seq_len == 0:
            self.seq_len = T
            
        preds_flat = preds.reshape(-1, F)  # (B*T, F)
        target_flat = target.reshape(-1, F)  # (B*T, F)
        mask_flat = mask.reshape(-1)  # (B*T,)
        
        # Only evaluate on masked positions
        masked_preds = preds_flat[mask_flat]  # (num_masked, F)
        masked_target = target_flat[mask_flat]  # (num_masked, F)
        
        if masked_preds.numel() == 0:  # No masked positions in this batch
            return
            
        # Update metrics
        self.mse.update(masked_preds, masked_target)
        self.mae.update(masked_preds, masked_target)
        
        # For PCC, we'll compute it per-feature and average
        for f in range(F):
            self.pcc.update(masked_preds[:, f], masked_target[:, f])
            
        # Track statistics
        self.num_samples += B
        self.total_masked += mask_flat.sum().item()
    
    def compute(self) -> Dict[str, Tensor]:
        if not hasattr(self, '_initialized') or not self._initialized:
            return {
                'MSE': torch.tensor(0.0),
                'MAE': torch.tensor(0.0),
                'PCC': torch.tensor(0.0),
                'mask_ratio': torch.tensor(0.0)
            }
            
        total_possible = self.num_samples * self.seq_len if hasattr(self, 'seq_len') and self.seq_len > 0 else 1
        mask_ratio = self.total_masked / total_possible if total_possible > 0 else 0.0
        
        # Get device from one of the metrics or use CPU
        device = 'cpu'
        if hasattr(self, 'mse') and self.mse is not None:
            device = next(iter(self.mse.buffers()), torch.tensor(0.0)).device
    
        results = {
            'MSE': self.mse.compute() if hasattr(self, 'mse') and self.mse is not None else torch.tensor(0.0),
            'MAE': self.mae.compute() if hasattr(self, 'mae') and self.mae is not None else torch.tensor(0.0),
            'PCC': self.pcc.compute() if hasattr(self, 'pcc') and self.pcc is not None and self.total_masked > 0 else torch.tensor(0.0),
            'mask_ratio': torch.tensor(mask_ratio, device=device)
        }
        self.reset()
        return results
    
    def reset(self):
        if hasattr(self, 'mse') and self.mse is not None:
            self.mse.reset()
        if hasattr(self, 'mae') and self.mae is not None:
            self.mae.reset()
        if hasattr(self, 'pcc') and self.pcc is not None:
            self.pcc.reset()
        self.num_samples = 0
        self.total_masked = 0
        torch.cuda.empty_cache()

class RunningMetrics:
    """Wrapper to manage multiple metrics based on task configuration"""
    def __init__(self, tasks_config: dict):
        self.metrics = {}
        
        for task_name, task_info in tasks_config.items():
            if task_name == 'sentiment':
                self.metrics[task_name] = SentimentMetrics()
            elif task_name in ['valence', 'arousal']:
                self.metrics[task_name] = ValenceArousalMetrics()
            elif task_name in ['emotion_class', 'emotion_class_fw', 'emotion_intensity']:
                self.metrics[task_name] = ClassificationMetrics(
                    num_classes=task_info['num_classes'],
                    average='macro',
                    ignore_index=-100
                )
            elif task_name in ['tmm_clip', 'tmm_wavlm_baseplus', 'tmm_xml_roberta']:
                self.metrics[task_name] = ReconstructionMetrics()
    
    def update(self, task_name: str, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None):
        if task_name in self.metrics:
            self.metrics[task_name].update(preds, target, mask)
    
    def compute(self) -> Dict[str, Dict[str, Tensor]]:
        results = {}
        for task_name, metric in self.metrics.items():
            results[task_name] = metric.compute()
        return results
    
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()


if __name__ == '__main__':
    import numpy as np
    from pprint import pprint
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Example configuration matching the YAML structure
    tasks_config = {
        'emotion_class': {
            'num_classes': 8,
            'metrics': ['ACC', 'P', 'R', 'F1']
        },
        'emotion_class_fw': {
            'num_classes': 8,
            'metrics': ['ACC', 'P', 'R', 'F1']
        },
        'emotion_intensity': {
            'num_classes': 3,
            'metrics': ['ACC', 'P', 'R', 'F1']
        },
        'sentiment': {
            'metrics': ['ACC_2', 'F1_2', 'ACC_7', 'F1_7', 'MAE', 'CORR']
        },
        'valence': {
            'metrics': ['CCC', 'RMSE', 'SAGR', 'PCC']
        },
        'arousal': {
            'metrics': ['CCC', 'RMSE', 'SAGR', 'PCC']
        }
    }
    
    def test_emotion_classification():
        print("\n=== Testing Emotion Classification (B, C) ===")
        metrics = RunningMetrics({"emotion_class": tasks_config['emotion_class']})
        batch_size = 16
        num_classes = 8
        
        # Generate more meaningful test data
        logits = torch.randn(batch_size, num_classes) * 2  # More spread out logits
        targets = torch.randint(0, num_classes, (batch_size,))
        mask = torch.ones(batch_size, dtype=torch.bool)
        mask[0] = 0  # Mask out first sample
        
        # Update multiple times to get better statistics
        for _ in range(10):
            metrics.update('emotion_class', logits, targets, mask)
        
        results = metrics.compute()['emotion_class']
        pprint({k: round(v.item(), 4) for k, v in results.items()})
        return results['ACC'] > 0  # Basic sanity check

    def test_frame_wise_classification():
        print("\n=== Testing Frame-wise Emotion Classification (B, T, C) ===")
        metrics = RunningMetrics({"emotion_class_fw": tasks_config['emotion_class_fw']})
        batch_size = 16
        seq_len = 10
        num_classes = 8
        
        logits = torch.randn(batch_size, seq_len, num_classes) * 2
        targets = torch.randint(0, num_classes, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[0, :] = 0  # Mask out first sequence
        mask[:, 0] = 0  # Mask out first frame
        
        for _ in range(10):
            metrics.update('emotion_class_fw', logits, targets, mask)
        
        results = metrics.compute()['emotion_class_fw']
        pprint({k: round(v.item(), 4) for k, v in results.items()})
        return results['ACC'] > 0

    def test_sentiment_analysis():
        print("\n=== Testing Sentiment Analysis ===")
        metrics = RunningMetrics({"sentiment": tasks_config['sentiment']})
        batch_size = 16
        
        # Generate more meaningful test data
        targets = torch.rand(batch_size, 1) * 6 - 3  # Values in [-3, 3]
        preds = targets + torch.randn_like(targets) * 0.5  # Add some noise
        mask = torch.ones(batch_size, dtype=torch.bool)
        mask[0] = 0  # Mask out first sample
        
        for _ in range(10):
            metrics.update('sentiment', preds, targets, mask)
        
        results = metrics.compute()['sentiment']
        pprint({k: round(v.item(), 4) for k, v in results.items()})
        return results['MAE'] < 2.0  # MAE should be reasonable

    def test_valence_arousal():
        print("\n=== Testing Valence/Arousal (frame-wise) ===")
        metrics_valence = RunningMetrics({"valence": tasks_config['valence']})
        metrics_arousal = RunningMetrics({"arousal": tasks_config['arousal']})
        batch_size = 16
        seq_len = 10
        
        # Generate more meaningful test data
        targets = torch.rand(batch_size, seq_len, 1) * 2 - 1  # Values in [-1, 1]
        preds = targets + torch.randn_like(targets) * 0.2  # Add some noise
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[0, :] = 0  # Mask out first sequence
        mask[:, 0] = 0  # Mask out first frame
        
        for _ in range(10):
            metrics_valence.update('valence', preds, targets, mask)
            metrics_arousal.update('arousal', preds, targets, mask)
        
        valence_results = metrics_valence.compute()['valence']
        arousal_results = metrics_arousal.compute()['arousal']
        
        print("Valence Metrics:")
        pprint({k: round(v.item(), 4) for k, v in valence_results.items()})
        print("\nArousal Metrics:")
        pprint({k: round(v.item(), 4) for k, v in arousal_results.items()})
        
        return (not any(torch.isnan(torch.tensor(list(valence_results.values()))))) and \
               (not any(torch.isnan(torch.tensor(list(arousal_results.values())))))

    def test_metric_accumulation():
        print("\n=== Testing Metric Accumulation ===")
        metrics_config = {
            'emotion_intensity': {
                'num_classes': 3,
                'metrics': ['ACC', 'P', 'R', 'F1']
            }
        }
        
        # First test case
        metrics1 = RunningMetrics({"emotion_intensity": metrics_config['emotion_intensity']})
        preds1 = torch.tensor([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2], [0.1, 0.4, 1.9]])
        targets1 = torch.tensor([0, 1, 2])
        metrics1.update('emotion_intensity', preds1, targets1)
        
        # Second test case
        metrics2 = RunningMetrics(metrics_config)
        preds2 = torch.tensor([[0.2, 1.9, 0.1], [1.8, 0.3, 0.2]])
        targets2 = torch.tensor([1, 0])
        metrics2.update('emotion_intensity', preds2, targets2)
        
        # Combined test case
        metrics_combined = RunningMetrics(metrics_config)
        combined_preds = torch.cat([preds1, preds2])
        combined_targets = torch.cat([targets1, targets2])
        metrics_combined.update('emotion_intensity', combined_preds, combined_targets)
        
        # Get results
        results1 = metrics1.compute()['emotion_intensity']
        results2 = metrics2.compute()['emotion_intensity']
        results_combined = metrics_combined.compute()['emotion_intensity']
        
        # Manually combine results (should match combined_results)
        acc_combined = (results1['ACC'] * 3 + results2['ACC'] * 2) / 5
        
        print("Batch 1 results:")
        pprint({k: round(v.item(), 4) for k, v in results1.items()})
        print("\nBatch 2 results:")
        pprint({k: round(v.item(), 4) for k, v in results2.items()})
        print("\nCombined results:")
        pprint({k: round(v.item(), 4) for k, v in results_combined.items()})
        print(f"\nManually combined ACC: {acc_combined:.4f}")
        
        return abs(acc_combined - results_combined['ACC'].item()) < 1e-6

    def test_reconstruction_metrics():
        print("\n=== Testing Reconstruction Metrics ===")
        metrics = RunningMetrics({
            'tmm_clip': {'metrics': ['MSE', 'MAE', 'PCC', 'mask_ratio']}
        })
        
        batch_size = 8
        seq_len = 64
        feat_dim = 512  # Example feature dimension
        
        # Generate random features and predictions
        targets = torch.randn(batch_size, seq_len, feat_dim)
        preds = targets + torch.randn_like(targets) * 0.1  # Add some noise
        
        # Create a random mask (approximately 15% masked)
        mask = torch.rand(batch_size, seq_len) < 0.15
        
        # Update metrics
        for _ in range(5):  # Multiple batches
            metrics.update('tmm_clip', preds, targets, mask)
        
        results = metrics.compute()['tmm_clip']
        pprint({k: v.item() for k, v in results.items()})
        
        # Basic sanity checks
        return (results['MSE'] > 0 and 
                results['MAE'] > 0 and
                0 <= results['mask_ratio'] <= 1.0)

    # Run all tests
    tests_passed = 0
    tests = [
        test_emotion_classification,
        test_frame_wise_classification,
        test_sentiment_analysis,
        test_valence_arousal,
        test_metric_accumulation,
        test_reconstruction_metrics
    ]
    
    for test in tests:
        try:
            if test():
                print(f"✅ {test.__name__} passed")
                tests_passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {str(e)}")
    
    print(f"\n🎉 {tests_passed}/{len(tests)} tests passed!")