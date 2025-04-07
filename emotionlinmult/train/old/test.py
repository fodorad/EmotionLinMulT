from argparse import ArgumentParser
import pprint
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from emotionlinmult.train.train import load_config
from pytorch_lightning import seed_everything
import seaborn as sns
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from emotionlinmult.train.mead_dataset import MeadDataModule
from emotionlinmult.train.metrics import classification_metrics
from emotionlinmult.train.train import load_model
from exordium.utils.decorator import load_or_create


@load_or_create('pkl')
def inference(config: dict, model: torch.nn.Module, dataloader: DataLoader, **kwargs):
    device = f'cuda:{config.get("gpu_id", 0)}'
    model.to(device)

    y_true_ec_batch = []
    y_true_ei_batch = []
    y_score_ec_batch = []
    y_score_ei_batch = []

    for x, y_true in tqdm(dataloader, desc="Inference", total=len(dataloader)):

        if isinstance(x, dict): x = list(x.values()) # dict to list
        x = [inp.to(device) for inp in x]

        y_logit = model(x)
        y_logit_ec, y_logit_ei = y_logit
        y_prob_ec = F.softmax(y_logit_ec, dim=1) # (B, 8)
        y_prob_ei = F.softmax(y_logit_ei, dim=1) # (B, 3)
        y_true_ec = torch.squeeze(y_true['emotion_class'], dim=-1) # (B, 1) -> (B,)
        y_true_ei = torch.squeeze(y_true['emotion_intensity'], dim=-1) # (B, 1) -> (B,)
        y_score_ec = torch.squeeze(torch.argmax(y_prob_ec, dim=1), dim=-1) # (batch_size, 8) -> (batch_size,)
        y_score_ei = torch.squeeze(torch.argmax(y_prob_ei, dim=1), dim=-1) # (batch_size, 3) -> (batch_size,)

        y_true_ec_batch.append(y_true_ec.detach().cpu().numpy())
        y_true_ei_batch.append(y_true_ei.detach().cpu().numpy())
        y_score_ec_batch.append(y_score_ec.detach().cpu().numpy())
        y_score_ei_batch.append(y_score_ei.detach().cpu().numpy())

    y_true_ec_all  = np.concatenate(y_true_ec_batch, axis=0)
    y_true_ei_all  = np.concatenate(y_true_ei_batch, axis=0)
    y_score_ec_all = np.concatenate(y_score_ec_batch, axis=0)
    y_score_ei_all = np.concatenate(y_score_ei_batch, axis=0)

    return {
        "y_true_ec_all": y_true_ec_all,
        "y_true_ei_all": y_true_ei_all,
        "y_score_ec_all": y_score_ec_all,
        "y_score_ei_all": y_score_ei_all
    }


def test(config: dict):
    device = f"cuda:{config.get('gpu_id', 0)}"
    log_dir = Path(config['log_dir'])
    test_path = log_dir / "test"
    test_path.mkdir(parents=True, exist_ok=True)

    print(f"Test inference")
    #
    # create model
    #
    try:
        model_weights = next(log_dir.glob("**/" + "best_val_f1_ec.ckpt")) # F1_ec
    except Exception:
        raise ValueError(f'Missing weights at {log_dir}')

    assert model_weights is not None and Path(model_weights).exists()
    model_ec = load_model(config).to(device)
    state_dict = torch.load(model_weights, map_location=device)['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model_ec.load_state_dict(state_dict)
    model_ec.eval()
    print(f"{config.get('model_type', 'linmult')} weights are loaded to {device}: {model_weights}")
    '''
    try:
        model_weights = next(log_dir.glob("**/" + "best_val_loss.ckpt")) # F1_ei
    except Exception:
        raise ValueError(f'Missing weights at {log_dir}')

    assert model_weights is not None and Path(model_weights).exists()
    model_ei = load_model(config).to(device)
    state_dict = torch.load(model_weights, map_location=device)['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model_ei.load_state_dict(state_dict)
    model_ei.eval()
    print(f"{config.get('model_type', 'linmult')} weights are loaded to {device}: {model_weights}")
    '''
    #
    # create datamodule
    #
    datamodule = MeadDataModule(config)
    test_dataloader = datamodule.setup_and_test()
    print(f"MEAD test data is loaded.")

    output = inference(config, model_ec, test_dataloader, output_path=test_path / f"inference_ec.pkl")
    # output_ei = inference(config, model_ec, test_dataloader, output_path=test_path / f"inference_ei.pkl")

    metrics_ec = classification_metrics(output['y_true_ec_all'], output['y_score_ec_all'], n_classes=8)
    metrics_ei = classification_metrics(output['y_true_ei_all'], output['y_score_ei_all'], n_classes=3)

    # weighted presence, max aggregation presence, position
    base_name = ["Task"]
    metrics_name = ["Accuracy", "Precision", "Recall", "F1"]

    table = PrettyTable(base_name + metrics_name)
    table.add_row(["EC"] + [metrics_ec[metric_name] for metric_name in metrics_name])
    table.add_row(["EI"] + [metrics_ei[metric_name] for metric_name in metrics_name])
    print("\nTest:\n", table, sep="")

    with open(test_path / f"metrics.csv", "w", encoding="utf8", newline="") as f:
        f.write(table.get_csv_string())

    ncm_ec = metrics_ec["NormalizedConfusionMatrix"].reshape(8, 8)
    ncm_ec = np.round(ncm_ec, decimals=2)
    sns.heatmap(ncm_ec, annot=True, linewidths=2)
    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actuals", fontsize=18)
    plt.title("EC Normalized Confusion Matrix", fontsize=18)
    plt.savefig(str(test_path / "ncm_ec.png"))
    plt.close()

    ncm_ei = metrics_ei["NormalizedConfusionMatrix"].reshape(3, 3)
    ncm_ei = np.round(ncm_ei, decimals=2)
    sns.heatmap(ncm_ei, annot=True, linewidths=2)
    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actuals", fontsize=18)
    plt.title("EI Normalized Confusion Matrix", fontsize=18)
    plt.savefig(str(test_path / "ncm_ei.png"))
    plt.close()

    table_rows = []
    for camera_angle in ["top", "front", "down", "left_30", "left_60", "right_30", "right_60"]:
        test_dataloader = datamodule.setup_and_test_camera(camera_angle)
        print(f"MEAD test data (only {camera_angle}) is loaded.")
        output = inference(config, model_ec, test_dataloader, output_path=test_path / f"inference_{camera_angle}.pkl")
        metrics_ec = classification_metrics(output['y_true_ec_all'], output['y_score_ec_all'], n_classes=8)
        metrics_ei = classification_metrics(output['y_true_ei_all'], output['y_score_ei_all'], n_classes=3)
        table_rows.append([camera_angle, metrics_ec["F1"], metrics_ei["F1"]])

    # weighted presence, max aggregation presence, position
    base_name = ["Camera angle"]
    metrics_name = ["F1_ec", "F1_ei"]

    table_camera = PrettyTable(base_name + metrics_name)
    for row in table_rows:
        table_camera.add_row(row)
    print("\nTest camera-wise:\n", table_camera, sep="")

    with open(test_path / f"metrics_camera.csv", "w", encoding="utf8", newline="") as f:
        f.write(table_camera.get_csv_string())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.log_dir) / 'csv_logger' / 'version_0' / 'hparams.yaml'
    config = load_config(config_path)
    config |= vars(args)

    print("Config:")
    pprint.pprint(config)

    seed_everything(config['seed'])
    test(config)