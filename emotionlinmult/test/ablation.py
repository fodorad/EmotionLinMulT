import os
import argparse
import json
import numpy as np
from pprint import pprint
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv
from emotionlinmult.train.metrics import classification_metrics, to_numpy

from emotionlinmult.preprocess.MEAD.dataset import UNIFIED_EMOTION_CLASS_NAMES, UNIFIED_EMOTION_INTENSITY_NAMES
from emotionlinmult.preprocess.MEAD.dataset import create_dataset as create_mead_dataset
from emotionlinmult.preprocess.MEAD.create_webdataset import CAMERA_POSITIONS
from emotionlinmult.test.results import DATASET_CONFIG
from emotionlinmult.train.datamodule import MultiDatasetModule
from linmult import LinMulT, LinT, load_config


def get_dataloader(
        camera_id: int | None = None,
        drop_feature_list: list[str] | None = None,
        mode: str = 'avt',
    ):

    if mode == 'avt':
        feature_list = ['wavlm_baseplus', 'clip', 'xml_roberta']
    elif mode == 'av':
        feature_list = ['wavlm_baseplus', 'clip']
    else:
        raise ValueError(f'Unsupported mode {mode}')

    config = {
        'feature_list': feature_list,
        'target_list': ['emotion_class', 'emotion_intensity'], 
        'datasets': {
            'train': ['mead'],
            'valid': ['mead'],
            'test': ['mead']
        },
        'camera_id': camera_id,
        'drop_feature_list': drop_feature_list
    }

    test_dataset = create_mead_dataset("test", camera_id=camera_id, drop_feature_list=drop_feature_list)
    datamodule = MultiDatasetModule(config)
    test_dataloader = datamodule.wrap_dataset(dataset=test_dataset, subset="test", shuffle=False)

    print('\nDataloader is created with config:')
    pprint(config)

    return test_dataloader


def get_model(checkpoint_path: str, config_path: str):
    config = load_config(config_path)

    if config['model_name'] == 'LinT':
        model = LinT(config=config)
    elif config['model_name'] == 'LinMulT':
        model = LinMulT(config=config)
    else:
        raise ValueError(f'Unsupported model name {config["model_name"]}')

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device(f'cuda:{config.get("devices", [0])[0]}'))
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if 'awl' not in k}
    model.load_state_dict(state_dict)

    model.to('cuda:0')
    model.eval()

    print('\nModel is loaded with config:')
    pprint(config)

    return model


def _test_dataloader_camera_id():
    for camera_id in list(range(0, 7)) + [None]:
        test_dataloader = get_dataloader(camera_id=camera_id)
        for batch_ind, batch in enumerate(test_dataloader):
            print(f"\n{camera_id}, {list(batch['__key__'])[0]}: {batch['emotion_class'][0].item()}, {batch['emotion_intensity'][0].item()}")
            if batch_ind == 3: break


def _test_dataloader_feature_list():
    for drop_feature_list in [['clip'], ['wavlm_baseplus'], ['clip', 'wavlm_baseplus']]:
        test_dataloader = get_dataloader(drop_feature_list=drop_feature_list)
        for batch_ind, batch in enumerate(test_dataloader):
            print(f"\n{drop_feature_list}, {list(batch['__key__'])[0]}: {batch['emotion_class'][0].item()}, {batch['emotion_intensity'][0].item()}, {batch['clip'][0].shape}, {any(batch['wavlm_baseplus_mask'][0])}, {any(batch['clip_mask'][0])}")
            if batch_ind == 3: break


def create_results_table_camera(results: dict) -> PrettyTable:
    """
    Create a PrettyTable from the results dictionary.
    
    Args:
        results: Dictionary containing the evaluation results
        
    Returns:
        PrettyTable: Formatted table with the results
    """
    # Initialize the table
    table = PrettyTable()
    table.field_names = ["camera_id", "task", "set", "Support", "F1", "ACC"]

    for camera_id in ['all'] + list(range(0, 7)):
        camera_data = results[camera_id]

        # Add emotion_class results
        for intensity, metrics in camera_data['emotion_class'].items():
            support = metrics['Support']

            table.add_row([
                'all' if camera_id == 'all' else CAMERA_POSITIONS[camera_id],
                'emotion_class',
                'all' if intensity == 'all' else f'{UNIFIED_EMOTION_INTENSITY_NAMES[intensity]}',
                support,
                f"{float(metrics['F1']):.2f}",
                f"{float(metrics['ACC']):.2f}"
            ])

        # Add emotion_intensity results
        for emotion_class, metrics in camera_data['emotion_intensity'].items():
            support = metrics['Support']
            
            table.add_row([
                'all' if camera_id == 'all' else CAMERA_POSITIONS[camera_id],
                'emotion_intensity',
                'all' if emotion_class == 'all' else f'{UNIFIED_EMOTION_CLASS_NAMES[emotion_class]}',
                support,
                f"{float(metrics['F1']):.2f}",
                f"{float(metrics['ACC']):.2f}"
            ])

    table.align["camera_id"] = "r"
    table.align["task"] = "l"
    table.align["set"] = "l"
    table.align["Support"] = "r"
    table.align["F1"] = "r"
    table.align["ACC"] = "r"

    return table


def create_results_table_modality_dropout(results: dict) -> PrettyTable:
    """
    Create a PrettyTable from the results dictionary.
    
    Args:
        results: Dictionary containing the evaluation results
        
    Returns:
        PrettyTable: Formatted table with the results
    """
    # Initialize the table
    table = PrettyTable()
    table.field_names = ["modality", "task", "set", "Support", "F1", "ACC"]

    for modality in results.keys():
        modality_data = results[modality]

        # Add emotion_class results
        for intensity, metrics in modality_data['emotion_class'].items():
            support = metrics['Support']

            table.add_row([
                modality,
                'emotion_class',
                'all' if intensity == 'all' else f'{UNIFIED_EMOTION_INTENSITY_NAMES[intensity]}',
                support,
                f"{float(metrics['F1']):.2f}",
                f"{float(metrics['ACC']):.2f}"
            ])

        # Add emotion_intensity results
        for emotion_class, metrics in modality_data['emotion_intensity'].items():
            support = metrics['Support']
            
            table.add_row([
                modality,
                'emotion_intensity',
                'all' if emotion_class == 'all' else f'{UNIFIED_EMOTION_CLASS_NAMES[emotion_class]}',
                support,
                f"{float(metrics['F1']):.2f}",
                f"{float(metrics['ACC']):.2f}"
            ])

    table.align["modality"] = "r"
    table.align["task"] = "l"
    table.align["set"] = "l"
    table.align["Support"] = "r"
    table.align["F1"] = "r"
    table.align["ACC"] = "r"

    return table


def table_headpose(model):
    
    feature_list = ['wavlm_baseplus', 'clip', 'xml_roberta']
    target_list = ['emotion_class', 'emotion_intensity']

    results = {}
    for camera_id in ['all'] + list(range(0, 7)):
        test_dataloader = get_dataloader(camera_id=None if camera_id == 'all' else camera_id)

        y_score_emotion_class = []
        y_true_emotion_class = []

        y_score_emotion_intensity = []
        y_true_emotion_intensity = []

        for batch_ind, batch in enumerate(test_dataloader):
            x = [batch[feature_name].to('cuda:0') for feature_name in feature_list]
            x_masks = [batch[f'{feature_name}_mask'].to('cuda:0') for feature_name in feature_list]
            y_true = [batch[task].to('cuda:0') for task in target_list]
            y_true_masks = [batch[f'{task}_mask'].to('cuda:0') for task in target_list]

            preds_heads = model(x, x_masks)
            active_preds_heads = {task_name: preds_heads[task_name] for task_name in target_list}

            for i, task_name in enumerate(target_list):
                pred = active_preds_heads[task_name].detach().cpu().numpy()
                target = y_true[i].detach().cpu().numpy()
                mask = y_true_masks[i].detach().cpu().numpy()

                pred_score = np.argmax(pred, axis=1)
                assert all(mask), f"Mask is not all True for {task_name}"
 
                if task_name == 'emotion_class':
                    y_score_emotion_class.append(pred_score)
                    y_true_emotion_class.append(target)
                elif task_name == 'emotion_intensity':
                    y_score_emotion_intensity.append(pred_score)
                    y_true_emotion_intensity.append(target)

        y_score_emotion_class = np.concatenate(y_score_emotion_class, axis=0)
        y_true_emotion_class = np.concatenate(y_true_emotion_class, axis=0)

        y_score_emotion_intensity = np.concatenate(y_score_emotion_intensity, axis=0)
        y_true_emotion_intensity = np.concatenate(y_true_emotion_intensity, axis=0)

        results[camera_id] = {}
        results[camera_id]['emotion_class'] = {}
        results[camera_id]['emotion_intensity'] = {}

        # calculate emotion_class on all emotion_intensity classes:
        m = classification_metrics(y_score_emotion_class, y_true_emotion_class, n_classes=8)
        results[camera_id]['emotion_class']['all'] = {'Support': int(sum(m['Support'])), 'F1': round(float(m['F1']), 2), 'ACC': round(float(m['ACC']), 2)}
        # calculate emotion_class on emotion_intensity classes:
        for emotion_intensity in range(3):
            m = classification_metrics(y_score_emotion_class[y_true_emotion_intensity == emotion_intensity], y_true_emotion_class[y_true_emotion_intensity == emotion_intensity], n_classes=8)
            results[camera_id]['emotion_class'][emotion_intensity] = {'Support': int(sum(m['Support'])), 'F1': round(float(m['F1']), 2), 'ACC': round(float(m['ACC']), 2)}

        # calculate emotion_intensity on all emotion_class classes:
        m = classification_metrics(y_score_emotion_intensity, y_true_emotion_intensity, n_classes=3)
        results[camera_id]['emotion_intensity']['all'] = {'Support': int(sum(m['Support'])), 'F1': round(float(m['F1']), 2), 'ACC': round(float(m['ACC']), 2)}
        # calculate emotion_intensity on emotion_class classes:
        for emotion_class in range(8):
            m = classification_metrics(y_score_emotion_intensity[y_true_emotion_class == emotion_class], y_true_emotion_intensity[y_true_emotion_class == emotion_class], n_classes=3)
            results[camera_id]['emotion_intensity'][emotion_class] = {'Support': int(sum(m['Support'])), 'F1': round(float(m['F1']), 2), 'ACC': round(float(m['ACC']), 2)}

    output_path = Path(args.checkpoint_path).parents[1] / 'tables' / 'mead_headpose.json'
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")

    print("\nResults Summary:")
    print(create_results_table_camera(results))


def modality_dropout(model, mode: str = 'avt'):

    if mode == 'avt':
        feature_list = ['wavlm_baseplus', 'clip', 'xml_roberta']
        drop_feature_list_options = [['wavlm_baseplus', 'clip', 'xml_roberta'], ['wavlm_baseplus', 'clip'], ['wavlm_baseplus', 'xml_roberta'], ['clip', 'xml_roberta']]
    elif mode == 'av':
        feature_list = ['wavlm_baseplus', 'clip']
        drop_feature_list_options = [['wavlm_baseplus', 'clip'], ['clip'], ['wavlm_baseplus']]
    else:
        raise ValueError(f'Unsupported mode {mode}')

    target_list = ['emotion_class', 'emotion_intensity']

    results = {}
    for drop_feature_list in drop_feature_list_options:
        test_dataloader = get_dataloader(camera_id=None, drop_feature_list=drop_feature_list)

        y_score_emotion_class = []
        y_true_emotion_class = []

        y_score_emotion_intensity = []
        y_true_emotion_intensity = []

        for batch_ind, batch in enumerate(test_dataloader):
            x = [batch[feature_name].to('cuda:0') for feature_name in feature_list]
            x_masks = [batch[f'{feature_name}_mask'].to('cuda:0') for feature_name in feature_list]
            y_true = [batch[task].to('cuda:0') for task in target_list]
            y_true_masks = [batch[f'{task}_mask'].to('cuda:0') for task in target_list]

            preds_heads = model(x, x_masks)
            active_preds_heads = {task_name: preds_heads[task_name] for task_name in target_list}

            for i, task_name in enumerate(target_list):
                pred = active_preds_heads[task_name].detach().cpu().numpy()
                target = y_true[i].detach().cpu().numpy()
                mask = y_true_masks[i].detach().cpu().numpy()

                pred_score = np.argmax(pred, axis=1)
                assert all(mask), f"Mask is not all True for {task_name}"
 
                if task_name == 'emotion_class':
                    y_score_emotion_class.append(pred_score)
                    y_true_emotion_class.append(target)
                elif task_name == 'emotion_intensity':
                    y_score_emotion_intensity.append(pred_score)
                    y_true_emotion_intensity.append(target)

        y_score_emotion_class = np.concatenate(y_score_emotion_class, axis=0)
        y_true_emotion_class = np.concatenate(y_true_emotion_class, axis=0)

        y_score_emotion_intensity = np.concatenate(y_score_emotion_intensity, axis=0)
        y_true_emotion_intensity = np.concatenate(y_true_emotion_intensity, axis=0)

        modality_name = ' + '.join(drop_feature_list)

        results[modality_name] = {}
        results[modality_name]['emotion_class'] = {}
        results[modality_name]['emotion_intensity'] = {}

        # calculate emotion_class on all emotion_intensity classes:
        m = classification_metrics(y_score_emotion_class, y_true_emotion_class, n_classes=8)
        results[modality_name]['emotion_class']['all'] = {'Support': int(sum(m['Support'])), 'F1': round(float(m['F1']), 2), 'ACC': round(float(m['ACC']), 2)}
        # calculate emotion_class on emotion_intensity classes:
        for emotion_intensity in range(3):
            m = classification_metrics(y_score_emotion_class[y_true_emotion_intensity == emotion_intensity], y_true_emotion_class[y_true_emotion_intensity == emotion_intensity], n_classes=8)
            results[modality_name]['emotion_class'][emotion_intensity] = {'Support': int(sum(m['Support'])), 'F1': round(float(m['F1']), 2), 'ACC': round(float(m['ACC']), 2)}

        # calculate emotion_intensity on all emotion_class classes:
        m = classification_metrics(y_score_emotion_intensity, y_true_emotion_intensity, n_classes=3)
        results[modality_name]['emotion_intensity']['all'] = {'Support': int(sum(m['Support'])), 'F1': round(float(m['F1']), 2), 'ACC': round(float(m['ACC']), 2)}
        # calculate emotion_intensity on emotion_class classes:
        for emotion_class in range(8):
            m = classification_metrics(y_score_emotion_intensity[y_true_emotion_class == emotion_class], y_true_emotion_intensity[y_true_emotion_class == emotion_class], n_classes=3)
            results[modality_name]['emotion_intensity'][emotion_class] = {'Support': int(sum(m['Support'])), 'F1': round(float(m['F1']), 2), 'ACC': round(float(m['ACC']), 2)}

    output_path = Path(args.checkpoint_path).parents[1] / 'tables' / 'mead_modality_dropout.json'
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")

    print("\nResults Summary:")
    print(create_results_table_modality_dropout(results))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script with improved argparse handling")
    parser.add_argument("--model_config_path", type=str, default="configs/MDMTL/stage2/model_40_AV.yaml", help="Path to the Model config file")
    parser.add_argument("--checkpoint_path", type=str, default="results/SD/stage2/mead_AV_ec_ei_40_all/checkpoint/checkpoint_valid_emotion_class_F1.ckpt", help="Path to the checkpoint file")
    parser.add_argument("--mode", type=str, default="avt", help="Mode for modality dropout")
    args = parser.parse_args()

    model = get_model(checkpoint_path=args.checkpoint_path, config_path=args.model_config_path)
    table_headpose(model)
    modality_dropout(model, mode=args.mode)
    exit()