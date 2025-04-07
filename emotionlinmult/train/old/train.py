import os, sys
from argparse import ArgumentParser
from pathlib import Path
import pickle
import datetime
import pprint
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from matplotlib import pyplot as plt
from linmult import LinMulT, LinT
from emotionlinmult.train.visualize import plot_loss, plot_f1, plot_ncm, plot_cm
from emotionlinmult.train.scheduler import CosineWarmupScheduler
from emotionlinmult.train.mead_dataset import MeadDataModule, load_config
from emotionlinmult.train.metrics import classification_metrics


class ModelWrapper(pl.LightningModule):

    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.device_str = f"cuda:{config['gpu_id']}"
        self.model = model
        self.model.to(self.device_str)
        self.criterion_ec = torch.nn.CrossEntropyLoss() # combines sigmoid and ce
        self.criterion_ei = torch.nn.CrossEntropyLoss()
        self.f1_emotion_class = torchmetrics.F1Score(num_classes=8)
        self.f1_emotion_intensity = torchmetrics.F1Score(num_classes=3)
        self.model_checkpoint = {
            "loss": float("inf"),
            "F1_ec": 0,
            "F1_ei": 0,
        }

        self.train_step_outputs = []
        self.valid_step_outputs = []
        self.test_step_outputs = []

        with open(config['history_ec_path'], "wb") as f:
            pickle.dump({'train': [], 'valid': [], 'test': []}, f)

        with open(config['history_ei_path'], "wb") as f:
            pickle.dump({'train': [], 'valid': [], 'test': []}, f)

        if config.get('model_weights', None):
            self.load_model(config["model_weights"])

        self.save_hyperparameters(config)


    def load_model(self, weights_path: str):
        self.model.to(self.device_str)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device_str), strict=False)


    def forward(self, x):
        if isinstance(x, dict):
            _x = list(x.values()) # dict to list
        return self.model(_x)


    def shared_step(self, batch, batch_idx):
        x, y_true = batch

        # x is dict of M features with shape (batch_size, time_dim, feature_dim)
        # x[feature_name].shape == (B, T, F)
        # y is dict of emotion classes and intensities with shape (batch_size, 1) and (batch_size, 1)
        # y['emotion_class'].shape == (B,1) and y['emotion_intensity'].shape == (B,1)
        # cls_logit[0].shape == (B,8) and cls_logit[1].shape == (B,3)
        y_logit = self.forward(x)
        y_logit_ec, y_logit_ei = y_logit
        y_prob_ec = F.softmax(y_logit_ec, dim=1) # (B, 8)
        y_prob_ei = F.softmax(y_logit_ei, dim=1) # (B, 3)
        y_true_ec = torch.squeeze(y_true['emotion_class'], dim=-1) # (B, 1) -> (B,)
        y_true_ei = torch.squeeze(y_true['emotion_intensity'], dim=-1) # (B, 1) -> (B,)
        loss_ec = self.criterion_ec(y_logit_ec, y_true_ec) # logits (B, 8) and labels (B,)
        loss_ei = self.criterion_ei(y_logit_ei, y_true_ei) # logits (B, 3) and labels (B,)
        weight_ec = 0.5
        weight_ei = 0.5
        loss = weight_ec * loss_ec + weight_ei * loss_ei
        y_score_ec = torch.squeeze(torch.argmax(y_prob_ec, dim=1), dim=-1) # (batch_size, 8) -> (batch_size,)
        y_score_ei = torch.squeeze(torch.argmax(y_prob_ei, dim=1), dim=-1) # (batch_size, 3) -> (batch_size,)
        return {
            'loss': loss,
            'loss_ec': loss_ec,
            'loss_ei': loss_ei,
            'y_true_ec': y_true_ec,
            'y_true_ei': y_true_ei,
            'y_logit_ec': y_logit_ec,
            'y_logit_ei': y_logit_ei,
            'y_prob_ec': y_prob_ec,
            'y_prob_ei': y_prob_ei,
            'y_score_ec': y_score_ec,
            'y_score_ei': y_score_ei,
        }


    def training_step(self, batch, batch_idx):
        batch_dict = self.shared_step(batch, batch_idx)
        self.train_step_outputs.append(batch_dict)
        self.log_dict({"train_loss": batch_dict['loss']}, on_epoch=True, batch_size=config['batch_size'])

        f1_score_ec = self.f1_emotion_class(batch_dict['y_prob_ec'], batch_dict['y_true_ec']) # probs (B, 8) and labels (B,)
        f1_score_ei = self.f1_emotion_intensity(batch_dict['y_prob_ei'], batch_dict['y_true_ei']) # probs (B, 3) and labels (B,)

        self.log("train_f1_ec", f1_score_ec, prog_bar=True, on_step=True, on_epoch=True, batch_size=config['batch_size'])
        self.log("train_f1_ei", f1_score_ei, prog_bar=True, on_step=True, on_epoch=True, batch_size=config['batch_size'])
        batch_dict |= {
            "train_f1_ec": f1_score_ec,
            "train_f1_ei": f1_score_ei,
        }
        return batch_dict


    def validation_step(self, batch, batch_idx):
        batch_dict = self.shared_step(batch, batch_idx)
        self.valid_step_outputs.append(batch_dict)

        self.log_dict({"val_loss": batch_dict['loss']}, on_epoch=True, batch_size=config['batch_size'])

        f1_score_ec = self.f1_emotion_class(batch_dict['y_prob_ec'], batch_dict['y_true_ec']) # probs (B, 8) and labels (B,)
        f1_score_ei = self.f1_emotion_intensity(batch_dict['y_prob_ei'], batch_dict['y_true_ei']) # probs (B, 3) and labels (B,)

        self.log("val_f1_ec", f1_score_ec, prog_bar=True, on_step=True, on_epoch=True, batch_size=config['batch_size'])
        self.log("val_f1_ei", f1_score_ei, prog_bar=True, on_step=True, on_epoch=True, batch_size=config['batch_size'])
        batch_dict |= {
            "val_f1_ec": f1_score_ec,
            "val_f1_ei": f1_score_ei,
        }
        return batch_dict


    def test_step(self, batch, batch_idx):
        batch_dict = self.shared_step(batch, batch_idx)
        self.test_step_outputs.append(batch_dict)
        self.log_dict({"test_loss": batch_dict['loss']}, on_epoch=True, batch_size=config['batch_size'])
        return batch_dict


    def save_model(self, metrics_ec: dict, metrics_ei: dict, output_dir: str | Path) -> None:
        if metrics_ec is None or metrics_ei is None: return

        output_dir = Path(output_dir)

        if metrics_ec["loss"] <= self.model_checkpoint["loss"]:
            path = output_dir / "model_best_loss.pt"
            optimizer_path = output_dir / "optimizer_best_loss.pt"
            print(f'loss decreased from {self.model_checkpoint["loss"]} to {metrics_ec["loss"]}\nModel saved to {path}')
            self.model_checkpoint["loss"] = metrics_ec["loss"]
            torch.save(self.model.state_dict(), path)
            torch.save(self.optimizer.state_dict(), optimizer_path)

        if metrics_ec["F1"] >= self.model_checkpoint["F1_ec"]:
            path = output_dir / "model_best_F1_ec.pt"
            optimizer_path = output_dir / "optimizer_best_F1_ec.pt"
            print(f'F1_ec increased from {self.model_checkpoint["F1_ec"]} to {metrics_ec["F1"]}\nModel saved to {path}')
            self.model_checkpoint["F1_ec"] = metrics_ec["F1"]
            torch.save(self.model.state_dict(), path)
            torch.save(self.optimizer.state_dict(), optimizer_path)

        if metrics_ei["F1"] >= self.model_checkpoint["F1_ei"]:
            path = output_dir / "model_best_F1_ei.pt"
            optimizer_path = output_dir / "optimizer_best_F1_ei.pt"
            print(f'F1_ei increased from {self.model_checkpoint["F1_ei"]} to {metrics_ei["F1"]}\nModel saved to {path}')
            self.model_checkpoint["F1_ei"] = metrics_ei["F1"]
            torch.save(self.model.state_dict(), path)
            torch.save(self.optimizer.state_dict(), optimizer_path)


    def calculate_metrics(self, step_outputs, phase: str):
        y_true_ec = (torch.cat([d["y_true_ec"] for d in step_outputs], dim=0).detach().cpu().squeeze().numpy())
        y_true_ei = (torch.cat([d["y_true_ei"] for d in step_outputs], dim=0).detach().cpu().squeeze().numpy())
        y_score_ec = (torch.cat([d["y_score_ec"] for d in step_outputs], dim=0).detach().cpu().squeeze().numpy())
        y_score_ei = (torch.cat([d["y_score_ei"] for d in step_outputs], dim=0).detach().cpu().squeeze().numpy())
        mean_loss_ec = (torch.stack([d["loss_ec"] for d in step_outputs]).mean().detach().cpu().numpy())
        mean_loss_ei = (torch.stack([d["loss_ei"] for d in step_outputs]).mean().detach().cpu().numpy())
        mean_loss = (torch.stack([d["loss"] for d in step_outputs]).mean().detach().cpu().numpy())

        metrics_ec = classification_metrics(y_true_ec, y_score_ec, n_classes=8)
        metrics_ei = classification_metrics(y_true_ei, y_score_ei, n_classes=3)

        metrics_ec["loss"] = mean_loss
        metrics_ec["loss_ec"] = mean_loss_ec
        metrics_ec["loss_ei"] = mean_loss_ei

        metrics_ei["loss"] = mean_loss
        metrics_ei["loss_ec"] = mean_loss_ec
        metrics_ei["loss_ei"] = mean_loss_ei

        with open(config['history_ec_path'], "rb") as f:
            history_ec = pickle.load(f)

        with open(config['history_ei_path'], "rb") as f:
            history_ei = pickle.load(f)

        history_ec[phase].append(metrics_ec)
        history_ei[phase].append(metrics_ei)

        # plots weighted loss
        plot_loss(history_ec, f"{Path(config['history_ec_path']).parent}/plots/loss.png")

        # ec plots
        plot_f1(history_ec,  f"{Path(config['history_ec_path']).parent}/plots/ec_f1.png")
        plot_cm(history_ec,  f"{Path(config['history_ec_path']).parent}/plots/ec_f1_cm.png", 8, "F1")
        plot_ncm(history_ec, f"{Path(config['history_ec_path']).parent}/plots/ec_f1_ncm.png", 8, "F1")

        # ei plots
        plot_f1(history_ei,  f"{Path(config['history_ei_path']).parent}/plots/ei_f1.png")
        plot_cm(history_ei,  f"{Path(config['history_ei_path']).parent}/plots/ei_f1_cm.png", 3, "F1")
        plot_ncm(history_ei, f"{Path(config['history_ei_path']).parent}/plots/ei_f1_ncm.png", 3, "F1")

        with open(config['history_ec_path'], "wb") as f:
            pickle.dump(history_ec, f)
            print(f"History saved to {config['history_ec_path']}")

        with open(config['history_ei_path'], "wb") as f:
            pickle.dump(history_ei, f)
            print(f"History saved to {config['history_ei_path']}")

        print(f"\n\n[{phase}]:")
        print('Task metrics: Emotion Class')
        pprint.pprint(metrics_ec)
        print('Task metrics: Emotion Intensity')
        pprint.pprint(metrics_ei)

        return metrics_ec, metrics_ei


    def on_train_epoch_end(self):
        self.calculate_metrics(self.train_step_outputs, "train")
        self.train_step_outputs.clear()


    def on_validation_epoch_end(self):
        val_metrics_ec, val_metrics_ei = self.calculate_metrics(self.valid_step_outputs, "valid")
        self.save_model(val_metrics_ec, val_metrics_ei, config['checkpoint_dir'])
        print(f"Time: {datetime.datetime.now()}")
        self.valid_step_outputs.clear()


    def on_test_epoch_end(self):
        self.calculate_metrics(self.test_step_outputs, "test")
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        """Example:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr) # Default: weight_decay=0.01
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]
        """
        self.lr_scheduler = None

        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])

        elif config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

        elif config["optimizer"] == "radam":
            optimizer = torch.optim.RAdam(self.model.parameters(), lr=config["learning_rate"])

        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=config["weight_decay"])
        else:
            raise ValueError("Optimizer is not set.")

        if config["lr_scheduler"] == "cosine_warmup":
            # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
            self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=config["warmup"], max_iters=config["max_iters"])
            return {"optimizer": optimizer}

        if config["lr_scheduler"] == "warm_restart":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, verbose=True, last_epoch=config.get("resume_epoch", -1))
        else:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": "val_loss",
            },
        }


    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        if config['lr_scheduler'] == "cosine_warmup":
            self.lr_scheduler.step()  # Step per iteration


def load_model(config: dict) -> nn.Module:
    model_type = config.get('model_type', 'linmult')
    if model_type == 'linmult':
        model = LinMulT(config["config_model"])
    elif model_type == 'lint':
        model = LinT(config["config_model"])
    else:
        raise NotImplementedError(f'{model_type} is not implemented')
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config_dataloader", type=str, default="configs/dataloader_1i_au.yaml")
    parser.add_argument("--config_model", type=str, default="configs/model_1i_2o_V_linear.yaml")
    parser.add_argument("--config_train", type=str, default="configs/train_base.yaml")
    parser.add_argument("--experiment_id", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)

    def update_config_with_args(config, args):
        for key, value in vars(args).items():
            if key not in config or args.__getattribute__(key) != parser.get_default(key):
                config[key] = value
        return config

    args = parser.parse_args()
    config_args = vars(args) # dict
    config_args["root_dir"] = os.path.dirname(os.path.realpath(__file__))
    config = load_config(config_args["config_dataloader"])
    config |= load_config(config_args["config_model"])
    config |= load_config(config_args["config_train"])
    config = update_config_with_args(config, args) # update the config dict if the user actually gave this value
    config["root_dir"] = os.path.dirname(os.path.realpath(__file__))
    return config


def run_trainer(config, model, datamodule):
    config['save_dir'] = str(Path(config['base_save_dir']) / "{date:%Y-%m-%d_%H:%M:%S}{experiment_id}".format(date=datetime.datetime.now(), experiment_id='' if config['experiment_id'] == '' else '_' + config['experiment_id']))
    config['history_ec_path'] = str(Path(config['save_dir']) / "history_ec.pkl")
    config['history_ei_path'] = str(Path(config['save_dir']) / "history_ei.pkl")
    config['checkpoint_dir'] = str(Path(config['save_dir']) / "models")
    Path(config['base_save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    config['max_iters'] = config['max_epochs'] * len(datamodule.train_dataloader())
    model_wrapper = ModelWrapper(model=model, config=config)

    cp_val_loss_callback = ModelCheckpoint(monitor="val_loss", dirpath=config['checkpoint_dir'], filename="best_val_loss", mode="min")
    cp_val_f1_ec_callback = ModelCheckpoint(monitor="val_f1_ec", dirpath=config['checkpoint_dir'], filename="best_val_f1_ec", mode="max")
    cp_val_f1_ei_callback = ModelCheckpoint(monitor="val_f1_ei", dirpath=config['checkpoint_dir'], filename="best_val_f1_ei", mode="max")
    cp_last = ModelCheckpoint(dirpath=config['checkpoint_dir'], save_last=True)
    #early_stop_callback = EarlyStopping(monitor="val_f1_ec", patience=config['patience'], verbose=True, mode="max") # monitor="val_loss", mode="min"
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=config['patience'], verbose=True, mode="min") # monitor="val_loss", mode="min"
    csv_logger = loggers.CSVLogger(config['save_dir'], name="csv_logger")

    trainer = Trainer(
        default_root_dir=config['save_dir'],
        accelerator="gpu",
        devices=[config.get('gpu_id', 0)],
        callbacks=[cp_val_loss_callback, cp_val_f1_ec_callback, cp_val_f1_ei_callback, cp_last, early_stop_callback],
        max_epochs=config['max_epochs'],
        logger=[csv_logger],
        num_sanity_val_steps=0, # disable dataloader sanity check
        benchmark=True
    )

    if config.get('lr_finder', False):
        print("Running lr finder...")
        lr_finder = trainer.tuner.lr_find(model_wrapper, datamodule, num_training=200)
        fig = lr_finder.plot(suggest=True)
        plt.savefig(Path(config['save_dir']) / "lr_finder.png")
        print("Suggested learning rate:", lr_finder.suggestion())
        exit()

    print(f"Training is started at {datetime.datetime.now()}")
    print("config:")
    pprint.pprint(config)
    trainer.fit(model_wrapper, datamodule, ckpt_path=config.get('resume_from_checkpoint', None))

    del trainer, model_wrapper
    print(f"Training is finished at {datetime.datetime.now()}")


if __name__ == "__main__":
    print("CMD:\npython", " ".join(sys.argv))
    config = parse_args()
    seed_everything(config.get("seed", 42))
    model = load_model(config)
    datamodule = MeadDataModule(config["config_dataloader"]).setup("fit")
    run_trainer(config, model, datamodule)
