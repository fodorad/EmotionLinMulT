from pathlib import Path
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class History:

    def __init__(self):
        self.history = {}


    def save(self, output_dir: str | Path):
        """Save the history to a JSON file."""
        output_path = Path(output_dir) / "history.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return obj.item()  # Convert numpy scalars to Python scalars
            return obj  # Default case

        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=4, default=convert_to_serializable)
        print(f"\nHistory saved to {output_path}")


    def save_test(self, output_dir: str | Path):
        """Save the test results to a JSON file."""
        output_path = Path(output_dir) / "history_test.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return obj.item()  # Convert numpy scalars to Python scalars
            return obj  # Default case

        history_test = {'test': self.history['test']}
        
        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=4, default=convert_to_serializable)
        print(f"\nHistory saved to {output_path}")


    def load(self, output_path: str):
        """Load the history from a JSON file."""
        with open(output_path, "r") as f:
            self.history = json.load(f)
        print(f"\nHistory loaded from {output_path}")


    def update(self, phase, task, metric, value, epoch):
        if phase not in self.history:
            self.history[phase] = {}
        if task not in self.history[phase]:
            self.history[phase][task] = {}
        if metric not in self.history[phase][task]:
            self.history[phase][task][metric] = []

        # Store the value and the epoch
        self.history[phase][task][metric].append((epoch, value))


    def get_metric(self, phase, task, metric):
        return self.history.get(phase, {}).get(task, {}).get(metric, [])


    def plot(self, task, metric, output_file: str):

        data = self.get_metric("valid", "all", "avg_loss")
        epochs, values = zip(*data) if data else ([], [])
        best_loss_epoch = epochs[int(torch.argmin(torch.tensor(values)))]

        for phase in ["train", "valid"]:
            data = self.get_metric(phase, task, metric)
            epochs, values = zip(*data) if data else ([], [])
            plt.plot(epochs, values, "*-", label=f"{phase} {metric}")

        data = self.get_metric("valid", task, metric)
        epochs, values = zip(*data) if data else ([], [])
        
        plt.plot(best_loss_epoch, values[best_loss_epoch], 'ro', markersize=5, label=f'{best_loss_epoch}: {np.round(values[best_loss_epoch], decimals=3)}')
        
        if metric in ['F1', 'ACC', 'P', 'R']:
            best_metric_epoch = epochs[int(torch.argmax(torch.tensor(values)))]
            plt.plot(best_metric_epoch, values[best_metric_epoch], 'go', markersize=5, label=f'{best_metric_epoch}: {np.round(values[best_metric_epoch], decimals=3)}')

        plt.title(f"Metrics for {task}")
        plt.xlabel("Epoch")
        plt.xticks(range(len(epochs)), [str(epoch) if epoch % 5 == 0 else "" for epoch in range(len(epochs))])

        if metric in ['F1', 'ACC', 'P', 'R']:
            plt.ylim([0, 1])
            plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylabel(metric)
        plt.legend(loc="upper left")

        plt.grid(True)
        plt.tight_layout()

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_file))
        plt.close()


    def plot_ncm(self, phase, task, metric, output_dir, n_classes: int):
        """Plot the Normalized Confusion Matrix (NCM)"""
        # Extract confusion matrices for the given phase and metric
        confusion_matrices = [epoch_ncm_tuple[1] for epoch_ncm_tuple in self.history[phase][task]["NormalizedConfusionMatrix"]]

        if phase == "test":
            # For test phase, expect only one entry
            if len(confusion_matrices) != 1:
                raise ValueError("Test history should contain exactly one entry for confusion matrix.")
            best_confusion_matrix = confusion_matrices[0]
            if best_confusion_matrix is None:
                raise ValueError("The test confusion matrix histroy is missing or invalid.")
        else:
            # For other phases like "valid", find the best confusion matrix index based on the specified metric
            if not confusion_matrices or all(cm is None for cm in confusion_matrices):
                raise ValueError("No valid confusion matrix history found for the given phase and metric.")

            metrics = [epoch_metric_tuple[1] for epoch_metric_tuple in self.history[phase][task][metric]]
            best_valid_index = np.nanargmax(metrics)  # Find the index of the best value

            # Select the best confusion matrix
            best_confusion_matrix = confusion_matrices[best_valid_index]
            if best_confusion_matrix is None:
                raise ValueError("The best confusion matrix history is missing or invalid.")

        # Reshape the confusion matrix
        best_confusion_matrix = np.array(best_confusion_matrix).reshape(n_classes, n_classes)

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            best_confusion_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            linewidths=0.5,
            xticklabels=[f"Class {i}" for i in range(n_classes)],
            yticklabels=[f"Class {i}" for i in range(n_classes)]
        )
        
        # Add labels and title
        plt.xlabel("Predicted Labels", fontsize=14)
        plt.ylabel("True Labels", fontsize=14)
        plt.title(f"Normalized Confusion Matrix ({phase.capitalize()}, {task}, {metric})", fontsize=16)

        # Save the plot
        output_path = output_dir / f'{phase}_{task}_{metric}_ncm.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()



    '''
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
    '''