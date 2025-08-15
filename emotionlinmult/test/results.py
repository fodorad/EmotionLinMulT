import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv
from emotionlinmult.train.metrics import classification_metrics, calculate_sentiment, calculate_va, to_numpy, save_ncm_plot

from emotionlinmult.preprocess.AffWild2.dataset_expr import UNIFIED_TO_EMOTION_CLASS as AFFWILD2_UNIFIED_TO_EMOTION_CLASS
from emotionlinmult.preprocess.CAER.dataset import UNIFIED_TO_EMOTION_CLASS as CAER_UNIFIED_TO_EMOTION_CLASS
from emotionlinmult.preprocess.CREMA_D.dataset import UNIFIED_TO_EMOTION_CLASS as CREMA_D_UNIFIED_TO_EMOTION_CLASS
from emotionlinmult.preprocess.MELD.dataset import UNIFIED_TO_EMOTION_CLASS as MELD_UNIFIED_TO_EMOTION_CLASS
from emotionlinmult.preprocess.MEAD.dataset import UNIFIED_TO_EMOTION_CLASS as MEAD_UNIFIED_TO_EMOTION_CLASS
from emotionlinmult.preprocess.RAVDESS.dataset import UNIFIED_TO_EMOTION_CLASS as RAVDESS_UNIFIED_TO_EMOTION_CLASS
from emotionlinmult.preprocess.CelebV_HQ.dataset import UNIFIED_TO_EMOTION_CLASS as CELEBV_HQ_UNIFIED_TO_EMOTION_CLASS

from emotionlinmult.preprocess.AffWild2.dataset_expr import EMOTION_CLASS_NAMES as AFFWILD2_EMOTION_CLASS_NAMES
from emotionlinmult.preprocess.RAVDESS.dataset import EMOTION_CLASS_NAMES as RAVDESS_EMOTION_CLASS_NAMES
from emotionlinmult.preprocess.MELD.dataset import EMOTION_CLASS_NAMES as MELD_EMOTION_CLASS_NAMES
from emotionlinmult.preprocess.MEAD.dataset import EMOTION_CLASS_NAMES as MEAD_EMOTION_CLASS_NAMES
from emotionlinmult.preprocess.CREMA_D.dataset import EMOTION_CLASS_NAMES as CREMA_D_EMOTION_CLASS_NAMES
from emotionlinmult.preprocess.CREMA_D.dataset import INTENSITY_CLASS_NAMES
from emotionlinmult.preprocess.CAER.dataset import EMOTION_CLASS_NAMES as CAER_EMOTION_CLASS_NAMES
from emotionlinmult.preprocess.CelebV_HQ.dataset import EMOTION_CLASS_NAMES as CELEBV_HQ_EMOTION_CLASS_NAMES

# Dataset configurations
DATASET_CONFIG = {
    'afew-va': {'tasks': ['valence', 'arousal'], 'metrics': ['CCC', 'RMSE', 'SAGR', 'PCC']},
    'affwild2_expr': {
        'tasks': {
            'emotion_class_fw': {'n_classes': 7, 'convert_dict': AFFWILD2_UNIFIED_TO_EMOTION_CLASS, 'class_names': AFFWILD2_EMOTION_CLASS_NAMES}
        },
        'metrics': ['ACC', 'P', 'R', 'F1', 'F1_macro']
    },
    'affwild2_va': {'tasks': ['valence', 'arousal'], 'metrics': ['CCC', 'RMSE', 'SAGR', 'PCC']},
    'caer': {
        'tasks': {
            'emotion_class': {'n_classes': 7, 'convert_dict': CAER_UNIFIED_TO_EMOTION_CLASS, 'class_names': CAER_EMOTION_CLASS_NAMES}
        },
        'combine': True,
        'metrics': ['ACC', 'P', 'R', 'F1', 'F1_macro']
    },
    'celebv-hq': {
        'tasks': {
            'emotion_class': {'n_classes': 8, 'convert_dict': CELEBV_HQ_UNIFIED_TO_EMOTION_CLASS, 'class_names': CELEBV_HQ_EMOTION_CLASS_NAMES}
        },
        'metrics': ['ACC', 'P', 'R', 'F1', 'F1_macro']
    },
    'crema-d': {
        'tasks': {
            'emotion_class': {'n_classes': 6, 'convert_dict': CREMA_D_UNIFIED_TO_EMOTION_CLASS, 'class_names': CREMA_D_EMOTION_CLASS_NAMES},
            'emotion_intensity': {'n_classes': 3, 'class_names': INTENSITY_CLASS_NAMES}
        },
        'metrics': ['ACC', 'P', 'R', 'F1', 'F1_macro']
    },
    'meld': {
        'tasks': {
            'emotion_class': {'n_classes': 7, 'convert_dict': MELD_UNIFIED_TO_EMOTION_CLASS, 'class_names': MELD_EMOTION_CLASS_NAMES}
        },
        'combine': True,
        'metrics': ['ACC', 'P', 'R', 'F1', 'F1_macro']
    },
    'mead': {
        'tasks': {
            'emotion_class': {'n_classes': 8, 'convert_dict': MEAD_UNIFIED_TO_EMOTION_CLASS, 'class_names': MEAD_EMOTION_CLASS_NAMES},
            'emotion_intensity': {'n_classes': 3, 'class_names': INTENSITY_CLASS_NAMES}
        },
        'combine': True,
        'metrics': ['ACC', 'P', 'R', 'F1', 'F1_macro']
    },
    'mosei': {'tasks': ['sentiment'], 'combine': True, 'metrics': ['ACC_2', 'ACC_7', 'F1_2', 'F1_7', 'MAE', 'CORR']},  # regression
    'ravdess': {
        'tasks': {
            'emotion_class': {'n_classes': 7, 'convert_dict': RAVDESS_UNIFIED_TO_EMOTION_CLASS, 'class_names': RAVDESS_EMOTION_CLASS_NAMES}
        },
        'combine': True,
        'metrics': ['ACC', 'P', 'R', 'F1', 'F1_macro']
    },
    'veatic': {'tasks': ['valence', 'arousal'], 'metrics': ['CCC', 'RMSE', 'SAGR', 'PCC']}
}


def load_predictions(pred_file: str) -> Dict[str, Dict]:
    """Load predictions from a JSON file.

    Expected format:
    {
        "task1": {
            "sample_id1": {"y_pred": [...], "y_true": value, "mask": bool},
            "sample_id2": {...}
        },
        "task2": {...}
    }
    """
    with open(pred_file, 'r') as f:
        return json.load(f)


def find_prediction_files(exp_path: Path) -> List[Path]:
    """Find all prediction files in an experiment directory."""
    files = list(exp_path.glob('*.json'))
    if not files: 
        raise ValueError(f"No prediction files found in {exp_path}")
    return files


def calculate_emotion_class(
        logits: np.ndarray | torch.Tensor,  # Shape: (N, 8) or (N, T, 8)
        targets: np.ndarray | torch.Tensor,  # Shape: (N,) or (N, T)
        mask: np.ndarray | torch.Tensor | None = None,  # Shape: (N,) or (N, T)
        is_framewise: bool = False,
        convert_dict: Optional[Dict[int, int]] = None,
        n_classes: int = 8,
    ) -> dict:
    """Calculate metrics for emotion classification task.

    Args:
        logits: Model predictions, shape (N, C) or (N, T, C) for C emotion classes
        targets: Target class indices, shape (N,) or (N, T) with values in [0, C-1]
        masks: Boolean mask for valid frames (N,) or (N, T), optional
        is_framewise: Whether the targets are framewise

    Returns:
        dict: Classification metrics dictionary
    """
    # Convert all inputs to numpy
    logits = to_numpy(logits)    # Shape: (N, C) or (N, T, C)
    targets = to_numpy(targets)  # Shape: (N,) or (N, T)

    if is_framewise:
        assert logits.ndim == 3 and targets.ndim == 2, \
            f"Expected logits and targets to be 3D and 2D arrays, got {logits.shape} and {targets.shape}"
        # Get predicted class indices
        preds = np.argmax(logits, axis=2)  # Shape: (N, T)
    else:
        assert logits.ndim == 2 and targets.ndim == 1, \
            f"Expected logits and targets to be 2D and 1D arrays, got {logits.shape} and {targets.shape}"
        # Get predicted class indices
        preds = np.argmax(logits, axis=1)  # Shape: (N,)

    if mask is not None:
        assert mask.ndim == 1 or mask.ndim == 2, \
            f"Expected mask to be 1D or 2D array, got {mask.shape}"

        mask = to_numpy(mask)      # Shape: (N,) or (N, T)
        preds = preds[mask]
        targets = targets[mask]
    
    if convert_dict is not None:
        preds = np.array([convert_dict.get(int(p), -1) for p in preds])
        targets = np.array([convert_dict.get(int(t), -1) for t in targets])

    return classification_metrics(preds, targets, n_classes=n_classes)


def save_dataset_task_table(dataset: str, task: str, result_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save multiple results for a dataset-task combination to a CSV file.
    
    Args:
        dataset: Name of the dataset
        task: Name of the task
        result_rows: List of dictionaries containing the result metrics
        output_dir: Directory to save the CSV file
    """
    if not result_rows:
        return
        
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a filename based on dataset and task
    filename = f"{dataset}_{task}.csv"
    output_file = output_dir / filename
    
    # Get all unique fieldnames from all result rows
    all_fieldnames = set()
    for row in result_rows:
        all_fieldnames.update(row.keys())
    
    # Define header fields that should always be included
    header_fields = ['setup', 'stage', 'experiment', 'checkpoint', 'dataset', 'task']
    
    # Filter fieldnames based on dataset config
    if dataset in DATASET_CONFIG and 'metrics' in DATASET_CONFIG[dataset]:
        # Include header fields and metrics that are in the dataset's metrics list
        metrics = DATASET_CONFIG[dataset]['metrics']
        filtered_fieldnames = header_fields + [f for f in all_fieldnames 
                                            if f in metrics and f not in header_fields]
    else:
        # If no specific metrics defined, include all fields
        filtered_fieldnames = list(all_fieldnames)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=filtered_fieldnames)
        writer.writeheader()
        for row in result_rows:
            # Only include fields that are in filtered_fieldnames
            writer.writerow({k: v for k, v in row.items() if k in filtered_fieldnames})
    
    # Create and print PrettyTable
    table = PrettyTable()
    table.field_names = filtered_fieldnames
    
    # Add all rows with formatted values
    for result_row in result_rows:
        row = []
        for field in filtered_fieldnames:
            value = result_row.get(field, '')
            # Format float values to 3 decimal places
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = f"{value:.3f}"
            row.append(value)
        table.add_row(row)
    
    print(f"\nResults for {dataset} - {task} (Total: {len(result_rows)} experiments):")
    print(table)
    print(f"Saved to: {output_file}")


def process_experiment(exp_path: Path) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Process a single experiment prediction directory.
    
    Args:
        exp_path: Path to the experiment directory containing prediction files
    """
    (root, setup, stage, exp_name, _, checkpoint) = exp_path.parts

    # Create output directory structure: results/setup/stage/exp-name/tables/checkpoint_name/
    output_dir = Path(root) / setup / stage / exp_name / 'tables' / checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = find_prediction_files(exp_path)

    # iterate over dataset prediction files
    results = []
    for pred_file in pred_files:
        dataset = pred_file.stem[5:]  # test_afew-va.json -> test_afew-va -> afew-va
        tasks = DATASET_CONFIG[dataset]['tasks']
        is_combine = DATASET_CONFIG[dataset].get('combine', False)
        data = load_predictions(pred_file)

        # iterate over the tasks
        for task in tasks:
            if task not in data:
                print(f"Task {task} not found in {pred_file}")
                continue

            task_preds = data[task]

            header = {
                'setup': setup,
                'stage': stage,
                'experiment': exp_name,
                'checkpoint': checkpoint,
                'dataset': dataset,
                'task': task
            }

            # iterate over the samples
            preds_dict = {}
            targets_dict = {}
            masks_dict = {}
            for sample_id, sample_data in task_preds.items():
                if task in ['valence', 'arousal', 'emotion_class_fw']:
                    y_pred = np.array(list(sample_data['y_pred'].values()))
                    y_true = np.array(list(sample_data['y_true'].values()))
                    mask = np.array(list(sample_data['mask'].values()))
                else: # emotion_class, emotion_intensity, sentiment
                    y_pred = np.array(sample_data['y_pred'])
                    y_true = np.array(sample_data['y_true'])
                    mask = np.array(sample_data['mask'])

                if is_combine:
                    chunk_idx = sample_id.split('_')[-1] # 000
                    sample_id = sample_id[:-(len(chunk_idx)+1)] # _000 removed

                if sample_id not in preds_dict:
                    preds_dict[sample_id] = []
                    targets_dict[sample_id] = []
                    masks_dict[sample_id] = []

                preds_dict[sample_id].append(y_pred)
                targets_dict[sample_id].append(y_true)
                masks_dict[sample_id].append(mask)

            if is_combine:
                # combine the predictions
                for sample_id in preds_dict:
                    preds_dict[sample_id] = np.mean(preds_dict[sample_id], axis=0)
                    if not all([elem == targets_dict[sample_id][0] for elem in targets_dict[sample_id]]): 
                        raise ValueError(f"Not all targets are the same for sample {sample_id}")
                    if not all([elem == masks_dict[sample_id][0] for elem in masks_dict[sample_id]]): 
                        raise ValueError(f"Not all masks are the same for sample {sample_id}")
                    targets_dict[sample_id] = targets_dict[sample_id][0]
                    masks_dict[sample_id] = masks_dict[sample_id][0]
            else:
                for sample_id in preds_dict:
                    assert len(preds_dict[sample_id]) == 1, f"Multiple predictions for sample {sample_id}"
                    assert len(targets_dict[sample_id]) == 1, f"Multiple targets for sample {sample_id}"
                    assert len(masks_dict[sample_id]) == 1, f"Multiple masks for sample {sample_id}"
                    preds_dict[sample_id] = preds_dict[sample_id][0]
                    targets_dict[sample_id] = targets_dict[sample_id][0]
                    masks_dict[sample_id] = masks_dict[sample_id][0]

            # gather all data for metric calculation
            y_pred = np.array(list(preds_dict.values()))
            y_true = np.array(list(targets_dict.values()))
            mask = np.array(list(masks_dict.values()))

            if task == 'emotion_class':
                task_metrics = calculate_emotion_class(
                    logits=y_pred,
                    targets=y_true,
                    mask=mask,
                    is_framewise=False,
                    convert_dict=DATASET_CONFIG[dataset]['tasks'][task].get('convert_dict', None),
                    n_classes=DATASET_CONFIG[dataset]['tasks'][task]['n_classes']
                )
                save_ncm_plot(
                    ncm=np.array(task_metrics['NormalizedConfusionMatrix']).reshape(
                        DATASET_CONFIG[dataset]['tasks'][task]['n_classes'],
                        DATASET_CONFIG[dataset]['tasks'][task]['n_classes']
                    ), 
                    class_names=DATASET_CONFIG[dataset]['tasks'][task]['class_names'], 
                    output_dir=output_dir,
                    dataset_name=dataset, 
                    task_name=task
                )

            elif task == 'emotion_class_fw':
                task_metrics = calculate_emotion_class(
                    logits=y_pred,
                    targets=y_true,
                    mask=mask,
                    is_framewise=True,
                    convert_dict=DATASET_CONFIG[dataset]['tasks'][task].get('convert_dict', None),
                    n_classes=DATASET_CONFIG[dataset]['tasks'][task]['n_classes']
                )
                save_ncm_plot(
                    ncm=np.array(task_metrics['NormalizedConfusionMatrix']).reshape(
                        DATASET_CONFIG[dataset]['tasks'][task]['n_classes'],
                        DATASET_CONFIG[dataset]['tasks'][task]['n_classes']
                    ), 
                    class_names=DATASET_CONFIG[dataset]['tasks'][task]['class_names'], 
                    output_dir=output_dir, 
                    dataset_name=dataset, 
                    task_name=task
                )
            elif task == 'emotion_intensity':
                task_metrics = calculate_emotion_class(
                    logits=y_pred,
                    targets=y_true,
                    mask=mask,
                    is_framewise=False,
                    n_classes=DATASET_CONFIG[dataset]['tasks'][task]['n_classes']
                )
                save_ncm_plot(
                    ncm=np.array(task_metrics['NormalizedConfusionMatrix']).reshape(
                        DATASET_CONFIG[dataset]['tasks'][task]['n_classes'],
                        DATASET_CONFIG[dataset]['tasks'][task]['n_classes']
                    ), 
                    class_names=DATASET_CONFIG[dataset]['tasks'][task]['class_names'], 
                    output_dir=output_dir, 
                    dataset_name=dataset, 
                    task_name=task
                )
            elif task == 'sentiment':
                task_metrics = calculate_sentiment(
                    preds=y_pred,
                    targets=y_true,
                    mask=mask
                )
            elif task in ['valence', 'arousal']:
                task_metrics = calculate_va(
                    preds=y_pred,
                    targets=y_true,
                    mask=mask
                )
            else:
                task_metrics = {}

            if not task_metrics:
                print(f"Warning: No metrics calculated for {dataset} - {task}")
                continue

            # Create result row with header and metrics
            result_row = {**header, **task_metrics}

            # Add to results
            results.append((dataset, task, result_row))
    
    return results


def main():
    results_root = Path('results')

    # Find all experiment directories
    exp_paths = list(results_root.glob('*/*/*/predictions/*'))  # setup/stage1/exp_name/predictions/checkpoint
    if not exp_paths:
        print("No experiment directories found. Looking in:")
        print(f"  {results_root.absolute()}/setup/stage/experiment/predictions/checkpoint")
        return

    # Dictionary to store results grouped by dataset and task
    results_by_dataset_task = {}
    
    # Process each experiment
    for exp_path in exp_paths:
        if not exp_path.is_dir(): 
            continue
            
        print(f"\n{'='*100}")
        print(f"Processing experiment: {exp_path}")
        
        # Process the experiment
        exp_results = process_experiment(exp_path)
        
        # Group results by dataset and task
        for dataset, task, result_row in exp_results:
            key = (dataset, task)
            if key not in results_by_dataset_task:
                results_by_dataset_task[key] = []
            results_by_dataset_task[key].append(result_row)
    
    # Create a common output directory for all results
    output_dir = results_root / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results for each dataset-task combination
    for (dataset, task), result_rows in results_by_dataset_task.items():
        save_dataset_task_table(dataset, task, result_rows, output_dir)
    
    # Create and save summary table
    create_summary_table(results_by_dataset_task, output_dir)
    
    print("\nAll experiments processed successfully!")


def create_summary_table(results_by_dataset_task: dict, output_dir: Path) -> None:
    """Create a summary table with key metrics for all experiments.
    
    Args:
        results_by_dataset_task: Dictionary mapping (dataset, task) to list of result rows
        output_dir: Directory to save the summary table
    """
    # Define the main metric for each task type
    task_metrics = {
        'emotion_class': 'F1',
        'emotion_class_fw': 'F1',
        'emotion_intensity': 'F1',
        'sentiment': 'MAE',
        'valence': 'CCC',
        'arousal': 'CCC'
    }
    
    # List to collect all summary rows
    summary_rows = []
    
    # Process each dataset-task group
    for (dataset, task), result_rows in results_by_dataset_task.items():
        # Get the main metric for this task
        metric_key = None
        for task_pattern in task_metrics:
            if task_pattern in task:
                metric_key = task_metrics[task_pattern]
                break
        
        if not metric_key:
            print(f"Warning: No metric defined for task: {task}")
            continue
        
        # Process each experiment result
        for row in result_rows:
            # Find the metric value (handle case where it might not exist)
            metric_value = row.get(metric_key, None)
            if metric_value is None:
                # Try to find the metric in nested structure if it exists
                for key in row:
                    if isinstance(row[key], dict) and metric_key in row[key]:
                        metric_value = row[key][metric_key]
                        break
            
            if metric_value is not None:
                # Create a new row for this experiment-task-metric combination
                summary_rows.append({
                    'setup': row['setup'],
                    'stage': row['stage'],
                    'experiment': row['experiment'],
                    'checkpoint': row['checkpoint'],
                    'dataset': dataset,
                    'task': task,
                    'metric': metric_key,
                    'metric_value': metric_value
                })
    
    if not summary_rows:
        print("Warning: No valid metrics found for summary table")
        return
    
    # Create a DataFrame from the summary rows
    summary_df = pd.DataFrame(summary_rows)
    
    # Display pretty table
    from prettytable import PrettyTable
    
    # Create a copy of the dataframe for display
    display_df = summary_df.copy()
    
    # Format metric values to 3 decimal places
    if 'metric_value' in display_df.columns:
        display_df['metric_value'] = display_df['metric_value'].apply(
            lambda x: f"{x:.3f}" if pd.notnull(x) and isinstance(x, (int, float)) else x
        )
    
    # Create and configure the pretty table
    table = PrettyTable()
    
    # Set field names and alignment
    fields = ['setup', 'stage', 'experiment', 'checkpoint', 'dataset', 'task', 'metric', 'metric_value']
    table.field_names = fields
    
    # Set alignment (left for text, right for numbers)
    for field in fields:
        if field == 'metric_value':
            table.align[field] = 'r'  # Right align metric values
        else:
            table.align[field] = 'l'  # Left align all other fields
    
    # Add rows to the table
    for _, row in display_df.iterrows():
        table.add_row([row[field] for field in fields])
    
    # Print the table with a title
    print("\n" + "="*120)
    print("EXPERIMENT SUMMARY TABLE".center(120))
    print("="*120)
    print(table)
    print("="*120 + "\n")
    
    # Save to CSV
    summary_file = output_dir / 'experiment_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary table saved to: {summary_file}")


if __name__ == "__main__":
    main()
