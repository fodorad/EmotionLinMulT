from pathlib import Path
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt



def plot_sentiment_metrics(model, output_dir):
    """Saves the loss and metrics plots to the given directory."""

    if output_dir is None:
        print('The output_dir argument of plot_metrics is None. Skipping...')
        return

    (Path(output_dir) / 'visualization').mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find the epoch with the lowest validation mae
    best_epoch = int(torch.argmin(torch.tensor(model.valid_mae))) # Get the index of the highest validation 1-MAE value
    best_val_mae = model.valid_mae[best_epoch] 
    best_val_corr = model.valid_corr[best_epoch] 
    best_val_acc_2 = model.valid_acc_2[best_epoch] 
    best_val_acc_7 = model.valid_acc_7[best_epoch] 
    best_val_f1_2 = model.valid_f1_2[best_epoch] 
    best_val_f1_7 = model.valid_f1_7[best_epoch] 
    num_epochs = len(model.valid_mae)
    
    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.train_losses, "*-", label='Training Loss')
    plt.plot(model.valid_losses, "*-", label='Validation Loss')
    plt.plot(best_epoch, model.valid_losses[best_epoch], 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_loss.png')
    plt.savefig(plot_path)
    plt.close()

    # MAE plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.train_mae, "*-", label='Training MAE')
    plt.plot(model.valid_mae, "*-", label='Validation MAE')
    plt.plot(best_epoch, best_val_mae, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_mae.png')
    plt.savefig(plot_path)
    plt.close()

    # CORR plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_corr, "*-", label='Validation CORR')
    plt.plot(best_epoch, best_val_corr, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('Corr Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Corr')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_corr.png')
    plt.savefig(plot_path)
    plt.close()

    # Acc2 plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_acc_2, "*-", label='Validation Acc_2')
    plt.plot(best_epoch, best_val_acc_2, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('Acc2 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Acc2')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_acc_2.png')
    plt.savefig(plot_path)
    plt.close()

    # Acc7 plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_acc_7, "*-", label='Validation Acc_7')
    plt.plot(best_epoch, best_val_acc_7, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('Acc7 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Acc7')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_acc_7.png')
    plt.savefig(plot_path)
    plt.close()

    # F1_7 plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_f1_7, "*-", label='Validation F1_7')
    plt.plot(best_epoch, best_val_f1_7, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('F1_7 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1_7')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_F1_7.png')
    plt.savefig(plot_path)
    plt.close()

    # F1_2 plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_f1_2, "*-", label='Validation F1_2')
    plt.plot(best_epoch, best_val_f1_2, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('F1_2 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1_2')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_F1_2.png')
    plt.savefig(plot_path)
    plt.close()


def plot_loss(data, output_path):
    loss_train = [d["loss"] for d in data["train"]]
    loss_valid = [d["loss"] for d in data["valid"]]
    best_valid_loss_value = np.nanmin(loss_valid)
    best_valid_loss_index = loss_valid.index(best_valid_loss_value)
    plt.plot(range(len(loss_train)), loss_train, "b-*", label="train_loss")
    plt.plot(range(len(loss_valid)), loss_valid, "g-*", label="valid_loss")
    plt.plot(best_valid_loss_index, best_valid_loss_value, "ro")
    plt.xlim([0, len(loss_train)+1])
    plt.xticks(np.arange(0, len(loss_train), 1))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.title(f"Best valid loss at epoch {best_valid_loss_index}: {np.round(best_valid_loss_value, 3)}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_f1(data, output_path):
    f1_train = [d["F1"] for d in data["train"]]
    f1_valid = [d["F1"] for d in data["valid"]]
    best_valid_f1_value = np.nanmax(f1_valid)
    best_valid_f1_index = f1_valid.index(best_valid_f1_value)
    plt.plot(range(len(f1_train)), f1_train, "b-*", label="train_f1")
    plt.plot(range(len(f1_valid)), f1_valid, "g-*", label="valid_f1")
    plt.plot(best_valid_f1_index, best_valid_f1_value, "ro")
    plt.ylim([0, 1.01])
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlim([0, len(f1_valid)+1])
    plt.xticks(np.arange(0, len(f1_valid), 1))
    plt.xlabel("epochs")
    plt.ylabel("F1 score")
    plt.legend(loc="lower right")
    plt.title(f"Best valid F1 score at epoch {best_valid_f1_index}: {np.round(best_valid_f1_value, 3)}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

'''
def plot_f1_seq(data, output_path):
    f1_train = [d["F1"] for d in data["train"]]
    f1_valid = [d["F1"] for d in data["valid"]]
    best_valid_f1_value = np.nanmax(f1_valid)
    best_valid_f1_index = f1_valid.index(best_valid_f1_value)
    plt.plot(range(len(f1_train)), f1_train, "b-", label="train_f1")
    plt.plot(range(len(f1_valid)), f1_valid, "g-", label="valid_f1")
    plt.plot(best_valid_f1_index, best_valid_f1_value, "ro")
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel("epochs")
    plt.ylabel("F1 score")
    plt.legend(loc="lower right")
    plt.title(f"Best valid F1 score at epoch {best_valid_f1_index}: {np.round(best_valid_f1_value, 3)}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
'''

def plot_ncm(data, output_path, n_classes: int, metric: str = "F1"):
    valid = [d[metric] for d in data["valid"]]
    best_valid_index = valid.index(np.nanmax(valid))
    best_valid_confusion_matrix = data["valid"][best_valid_index]["NormalizedConfusionMatrix"].reshape(n_classes, n_classes)
    sns.heatmap(best_valid_confusion_matrix, annot=True, linewidths=2)
    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actuals", fontsize=18)
    plt.title("Normalized Confusion Matrix", fontsize=18)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_cm(data, output_path, n_classes: int,  metric: str = "F1"):
    valid = [d[metric] for d in data["valid"]]
    best_valid_index = valid.index(np.nanmax(valid))
    best_valid_confusion_matrix = data["valid"][best_valid_index]["ConfusionMatrix"].reshape(n_classes, n_classes)
    sns.heatmap(best_valid_confusion_matrix, annot=True, linewidths=2)
    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actuals", fontsize=18)
    plt.title("Confusion Matrix", fontsize=18)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()