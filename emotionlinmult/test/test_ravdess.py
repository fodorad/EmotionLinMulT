
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

