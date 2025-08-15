import torch
import json
import pprint
import webdataset as wds
import numpy as np
from pathlib import Path
from tqdm import tqdm
from emotionlinmult.preprocess import (
    CREMAD_EMOTION_MAPPING, 
    CREMAD_INTENSITY_MAPPING, 
    CREMAD_INTENSITY_NAME2ORIG
)
from emotionlinmult.preprocess.CREMAD import DB_PROCESSED


def _process_sample(sample):

    for key in ["emotion_class.npy", "clip.npy", "wavlm_baseplus.npy"]:
        if key in sample:
            sample[key] = torch.from_numpy(sample[key].copy())

    # convert emotion class to unified emotion class
    sample['emotion_class.npy'] = torch.tensor(CREMAD_EMOTION_MAPPING.to_unified_id(sample['emotion_class.npy'].item()), dtype=torch.int64)

    # handle masks
    mask_keys = [key for key in sample.keys() if key.endswith("_mask.npy")]
    for key in mask_keys:
        sample[key] = torch.from_numpy(sample[key].copy()).bool()

    sample['dataset'] = 'CREMA-D_expr'
    sample['emotion_class_mask.npy'] = torch.tensor(True, dtype=torch.bool)
    return sample


def get_shard_urls(subset: str):
    webdataset_dir = DB_PROCESSED / "webdataset_expr"
    pattern = f"cremad_{subset}_*.tar"
    shard_files = sorted(webdataset_dir.glob(pattern))
    urls = [str(f) for f in shard_files]
    if not urls: raise FileNotFoundError(f"No shards found for pattern {subset} in {webdataset_dir}")
    return urls


def create_dataset(subset: str, shuffle_buffer_size: int = 300):
    urls = get_shard_urls(subset)

    pipeline = [
        wds.WebDataset(urls, empty_check=False, shardshuffle=100 if subset == 'train' else False),
        wds.decode(),
        wds.map(_process_sample),
        wds.shuffle(shuffle_buffer_size if subset == 'train' else False),
    ]

    return wds.DataPipeline(*pipeline) 


def create_dataset_with_size(subset: str, epoch_size: int | None = None):
    wds_dataset = create_dataset(subset)
    if epoch_size is None:
        epoch_size = count_samples()[subset]
    wds_dataset = wds_dataset.with_epoch(epoch_size).with_length(epoch_size)
    return wds_dataset


def count_samples(cache_path: Path = DB_PROCESSED / "count_samples_expr.json", verbose: bool = False) -> dict[str, int]:
    if cache_path.exists():
        with open(cache_path, "r") as f:
            d = json.load(f)
    else:
        d = {subset: sum(1 for _ in create_dataset(subset)) for subset in ["train", "valid", "test"]}
        with open(cache_path, "w") as f:
            json.dump(d, f)
    if verbose: pprint.pprint(d)
    return d


def calculate_class_distribution():
    """Calculate and return class distribution statistics for CREMA-D expr dataset.
    
    Returns:
        dict: A dictionary containing counts for each subset (train/valid/test) with:
            - unified_emotion_class: Counts by unified emotion class name
            - dataset_emotion_class: Counts by original dataset emotion class name
            - unified_emotion_intensity: Counts by unified intensity name
            - dataset_emotion_intensity: Counts by original dataset intensity name
            - total: Total number of samples in the subset
    """
    from collections import defaultdict
    
    # Initialize results dictionary
    results = {}
    
    for subset in ['train', 'valid', 'test']:
        dataset = create_dataset(subset)
        
        # Initialize counters
        unified_emotion_counts = defaultdict(int)
        dataset_emotion_counts = defaultdict(int)
        subset_total = 0
        
        # Process samples
        for sample in tqdm(dataset, desc=f"Processing {subset} samples"):
            # Get unified class
            unified_class_id = int(sample['emotion_class.npy'])
            
            # Get unified names
            unified_emotion_name = CREMAD_EMOTION_MAPPING.unified_name(unified_class_id)

            # Get original dataset class and intensity (reverse mapping from unified to original)
            original_class_id = CREMAD_EMOTION_MAPPING.to_orig_id(unified_class_id)
            original_class_name = CREMAD_EMOTION_MAPPING.to_orig_name(original_class_id)
            unified_emotion_counts[unified_emotion_name] += 1
            dataset_emotion_counts[original_class_name] += 1
            subset_total += 1
            
        # Store results for this subset
        results[subset] = {
            'unified_emotion_class': dict(sorted(unified_emotion_counts.items())),
            'dataset_emotion_class': dict(sorted(dataset_emotion_counts.items())),
            'subset_total': subset_total,
        }

    # Print summary
    for subset in ['train', 'valid', 'test']:
        print(f"\n=== {subset.upper()} ===")
        print("Unified Emotion Classes:")
        pprint.pprint(results[subset]['unified_emotion_class'])
        print("\nDataset Emotion Classes:")
        pprint.pprint(results[subset]['dataset_emotion_class'])
        print(f"\nSubset total: {results[subset]['subset_total']}")
        print("=" * 50)

    # Save to file
    output_file = DB_PROCESSED / "class_distribution_expr.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nClass distribution saved to {output_file}")

    return results



def compute_class_weights():
    train_counts = get_class_distribution('train')
    # weights only for present classes (0,1,3,4,5,6), set 0 for absent class (2,7)!
    present_counts = train_counts[train_counts > 0]
    weights = torch.zeros(8, dtype=torch.float32)
    # Avoid division by zero for weights
    present_weights = 1.0 / (present_counts + 1e-6)
    # Normalize to mean=1 (over seen classes)
    present_weights = present_weights / present_weights.sum() * 6
    weights[:2] = present_weights[:2]  # Assign weights
    weights[3:7] = present_weights[2:]
    weights[2] = 0.0  # surprise is absent
    weights[7] = 0.0  # Contempt is absent
    return train_counts, weights


if __name__ == "__main__":

    #train_counts, class_weights = compute_class_weights()
    #print("Class counts:", train_counts)  # tensor([752, 880,   0, 878, 880, 880, 879,   0])
    #print("Class weights:", class_weights)  # tensor([1.1373, 0.9719, 0.0000, 0.9741, 0.9719, 0.9719, 0.9730, 0.0000])
    #exit()

    count_samples(verbose=True)
    calculate_class_distribution()
    exit()

    test_dataset = create_dataset("test")
    for i, sample in tqdm(enumerate(test_dataset), desc="Loading test samples"):
        print(f"[test] {i}")
        print(list(sample.keys()))
        print(sample['emotion_class.npy'].item(), sample['emotion_class_mask.npy'].item())
        print(sample['emotion_intensity.npy'].item(), sample['emotion_intensity_mask.npy'].item())
        if i == 10: break
