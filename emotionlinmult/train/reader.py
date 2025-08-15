import json
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple, Literal
import numpy as np
import torch


def load_predictions_json(
    predictions_path: Union[str, Path],
) -> Dict[str, Any]:
    """Load predictions from a JSON file."""
    predictions_path = Path(predictions_path)
    
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")
    
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    return predictions


def load_video_level_predictions(
    json_path: Union[str, Path], 
    task_name: Literal['sentiment', 'sentiment_class', 'emotion_intensity', 'emotion_class']
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load video-level predictions.
    
    Args:
        json_path: Path to the prediction JSON file
        task_name: Either 'sentiment', 'sentiment_class', 'emotion_intensity', or 'emotion_class'
        
    Returns:
        Dictionary mapping sample IDs to dicts with 'y_true' and 'y_pred' numpy arrays
    """
    data = load_predictions_json(json_path)[task_name]

    result = {}
    for sample_id, sample_data in data.items():
        result[sample_id] = {
            'y_true': np.array(sample_data['y_true'], dtype=np.float32),
            'y_pred': np.array(sample_data['y_pred'], dtype=np.float32)
        }

    return result


def load_frame_level_predictions(
    json_path: Union[str, Path], 
    task_name: Literal['valence', 'arousal', 'emotion_class_fw']
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load frame-level predictions.

    Args:
        json_path: Path to the prediction JSON file
        task_name: Either 'valence', 'arousal', or 'emotion_class_fw'
        
    Returns:
        Dictionary mapping sample IDs to dicts with 'y_true' and 'y_pred' numpy arrays
    """
    data = load_predictions_json(json_path)[task_name]
    result = {}

    for sample_id, sample_data in data.items():
        y_true = np.array([v for _, v in sample_data['y_true'].items()], dtype=np.float32)
        y_pred = np.array([v for _, v in sample_data['y_pred'].items()], dtype=np.float32)
        
        result[sample_id] = {
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    return result


if __name__ == '__main__':
    # Load valence and arousal predictions
    json_path = 'results/STL/va/20250605-001710/predictions/test_afew-va.json'
    valence = load_frame_level_predictions(json_path, 'valence')
    arousal = load_frame_level_predictions(json_path, 'arousal')
    
    # Load sentiment predictions
    json_path = 'results/STL/sentiment/20250530-190541/predictions/test_mosei.json'
    sentiment = load_video_level_predictions(json_path, 'sentiment')

    # Load sentiment class predictions
    json_path = 'results/STL/sentiment/20250530-190541/predictions/test_meld.json'
    sentiment_class = load_video_level_predictions(json_path, 'sentiment_class')

    # Load intensity predictions
    json_path = 'results/STL/emotion_intensity/20250610-145412/predictions/test_crema-d.json'
    intensity = load_video_level_predictions(json_path, 'emotion_intensity')
    
    # Load emotion class predictions
    json_path = 'results/STL/emotion_class/20250610-132847/predictions/test_celebv-hq.json'
    emotion_class = load_video_level_predictions(json_path, 'emotion_class')
    
    # Load emotion class predictions fw
    json_path = 'results/STL/emotion_class_fw/20250607-024313/predictions/test_affwild2_expr.json'
    emotion_class_fw = load_frame_level_predictions(json_path, 'emotion_class_fw')
    
    for taskname, data in {
        'valence': valence,
        'arousal': arousal,
        'sentiment': sentiment,
        'sentiment_class': sentiment_class,
        'emotion_class': emotion_class,
        'emotion_class_fw': emotion_class_fw,
        'emotion_intensity': intensity
    }.items():
        print(taskname, len(data), next(iter(data.values()))['y_true'].shape, next(iter(data.values()))['y_pred'].shape)
    