import yaml
import json
import numpy as np
from pathlib import Path


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_predictions(json_path):
    """Load predictions from a JSON file and convert values to numpy arrays.
    
    Args:
        json_path (str or Path): Path to the JSON file containing predictions
        
    Returns:
        dict: A dictionary where:
            - First level keys are dataset names
            - Second level keys are task names
            - Third level keys are sample keys
            - Fourth level contains 'y_pred' and 'y_true'
            Values are numpy arrays with format depending on task type:
            - sentiment: scalar float values
            - sentiment_class/emotion_class/intensity: array of logits for y_pred, scalar int for y_true
            - valence/arousal: dict of frame_id -> scalar float values
            - emotion_class_fw: dict of frame_id -> array of logits for y_pred, scalar int for y_true
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert values to numpy arrays
    for dataset, tasks in data.items():
        for task_name, samples in tasks.items():
            for sample_key, predictions in samples.items():
                if task_name == 'sentiment':
                    # Scalar float values
                    predictions['y_pred'] = np.array(predictions['y_pred'])
                    predictions['y_true'] = np.array(predictions['y_true'])
                elif task_name in ['sentiment_class', 'emotion_class', 'intensity']:
                    # Array of logits for pred, scalar int for true
                    predictions['y_pred'] = np.array(predictions['y_pred'])  # (C,) logits
                    predictions['y_true'] = np.array(predictions['y_true'])  # scalar class index
                elif task_name in ['valence', 'arousal']:
                    # Frame-wise scalar values
                    frame_ids = sorted(predictions['y_pred'].keys(), key=int)
                    pred_values = [predictions['y_pred'][fid] for fid in frame_ids]
                    true_values = [predictions['y_true'][fid] for fid in frame_ids]
                    predictions['y_pred'] = np.array(pred_values)  # (T,)
                    predictions['y_true'] = np.array(true_values)  # (T,)
                    predictions['frame_ids'] = np.array(frame_ids, dtype=int)  # Store original frame IDs
                elif task_name == 'emotion_class_fw':
                    # Frame-wise logits
                    frame_ids = sorted(predictions['y_pred'].keys(), key=int)
                    pred_values = [predictions['y_pred'][fid] for fid in frame_ids]  # list of logit arrays
                    true_values = [predictions['y_true'][fid] for fid in frame_ids]  # list of class indices
                    predictions['y_pred'] = np.array(pred_values)  # (T, C)
                    predictions['y_true'] = np.array(true_values)  # (T,)
                    predictions['frame_ids'] = np.array(frame_ids, dtype=int)  # Store original frame IDs
    
    return data