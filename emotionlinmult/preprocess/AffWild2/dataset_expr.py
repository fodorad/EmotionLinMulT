import torch
import json
import pprint
import webdataset as wds
import numpy as np
from pathlib import Path
from tqdm import tqdm
from emotionlinmult.preprocess import AFFWILD2_EMOTION_MAPPING
from emotionlinmult.preprocess.AffWild2 import DB_PROCESSED


def _process_sample(sample):

    for key in ["emotion_class_fw.npy", "clip.npy", "wavlm_baseplus.npy"]:
        if key in sample:
            sample[key] = torch.from_numpy(sample[key].copy())

    # convert emotion class to unified emotion class. the variable is actually a frame-wise label torch tensor with shape (T,)
    sample['emotion_class_fw.npy'] = torch.tensor([AFFWILD2_EMOTION_MAPPING.to_unified_id(int(label)) for label in sample['emotion_class_fw.npy']], dtype=torch.int64)

    # handle masks
    mask_keys = [key for key in sample.keys() if key.endswith("_mask.npy")]
    for key in mask_keys:
        sample[key] = torch.from_numpy(sample[key].copy()).bool()

    sample['dataset'] = 'AffWild2_expr'
    return sample


def get_shard_urls(subset: str):
    webdataset_dir = DB_PROCESSED / "webdataset_expr"
    pattern = f"affwild2_expr_{subset}_*.tar"
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
    """Calculate and return class distribution statistics for AffWild2 expr dataset.
    
    Returns:
        dict: A dictionary containing counts for each subset (train/valid/test) with:
            - unified_emotion_class: Counts by unified emotion class name
            - dataset_emotion_class: Counts by original dataset emotion class name
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
        unified_emotion_counts_video_level = defaultdict(int)
        dataset_emotion_counts_video_level = defaultdict(int)
        subset_total = 0
        
        # Process samples
        for sample in tqdm(dataset, desc=f"Processing {subset} samples"):
            # Get unified class
            unified_class_id = [int(elem) for elem in sample['emotion_class_fw.npy'] if elem != -1]
            
            # Get unified names
            unified_emotion_name = [AFFWILD2_EMOTION_MAPPING.unified_name(elem) for elem in unified_class_id]

            # Get original dataset class and intensity (reverse mapping from unified to original)
            original_class_id = [AFFWILD2_EMOTION_MAPPING.to_orig_id(elem) for elem in unified_class_id]
            original_class_name = [AFFWILD2_EMOTION_MAPPING.to_orig_name(elem) for elem in original_class_id]
            for elem in unified_emotion_name: unified_emotion_counts[elem] += 1
            for elem in original_class_name: dataset_emotion_counts[elem] += 1

            # determine which class the video has the most frequent, that will be the video level annotation:
            # unified_class_id this is a list of unified class ids, get the most frequent element
            # what is counter?
            from collections import Counter
            video_level_unified_class_id = Counter(unified_class_id).most_common(1)[0][0]
            video_level_unified_emotion_name = AFFWILD2_EMOTION_MAPPING.unified_name(video_level_unified_class_id)
            video_level_original_class_id = AFFWILD2_EMOTION_MAPPING.to_orig_id(video_level_unified_class_id)
            video_level_original_class_name = AFFWILD2_EMOTION_MAPPING.to_orig_name(video_level_original_class_id)

            unified_emotion_counts_video_level[video_level_unified_emotion_name] += 1
            dataset_emotion_counts_video_level[video_level_original_class_name] += 1
            subset_total += 1
            
        # Store results for this subset
        results[subset] = {
            'unified_emotion_class': dict(sorted(unified_emotion_counts.items())),
            'dataset_emotion_class': dict(sorted(dataset_emotion_counts.items())),
            'unified_emotion_class_video_level': dict(sorted(unified_emotion_counts_video_level.items())),
            'dataset_emotion_class_video_level': dict(sorted(dataset_emotion_counts_video_level.items())),
            'subset_total': subset_total,
        }

    # Print summary
    for subset in ['train', 'valid', 'test']:
        print(f"\n=== {subset.upper()} ===")
        print("Unified Emotion Classes:")
        pprint.pprint(results[subset]['unified_emotion_class'])
        print("\nDataset Emotion Classes:")
        pprint.pprint(results[subset]['dataset_emotion_class'])
        print("\nUnified Emotion Classes (video level):")
        pprint.pprint(results[subset]['unified_emotion_class_video_level'])
        print("\nDataset Emotion Classes (video level):")
        pprint.pprint(results[subset]['dataset_emotion_class_video_level'])
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
    # weights only for present classes (0–6), set 0 for absent class (7)!
    present_counts = train_counts[:7]
    weights = torch.zeros(8, dtype=torch.float32)
    # Avoid division by zero for weights
    present_weights = 1.0 / (present_counts + 1e-6)
    # Normalize to mean=1 (over seen classes)
    present_weights = present_weights / present_weights.sum() * 7
    weights[:7] = present_weights  # Assign weights
    weights[7] = 0.0  # Contempt is absent
    return train_counts, weights


if __name__ == "__main__":

    count_samples(verbose=True)
    #calculate_class_distribution()
    #exit()
    
    #train_counts, class_weights = compute_class_weights()
    #print("Class counts:", train_counts) # tensor([160947,  81604,  28725,  65138,  14293,   4498,   8840,      0])
    #print("Class weights:", class_weights) # tensor([0.0917, 0.1810, 0.5141, 0.2267, 1.0331, 3.2829, 1.6704, 0.0000])

    ds = create_dataset("train")
    for i, sample in tqdm(enumerate(ds), desc="Loading train samples"):
        print(f"[train] sample {i}")
        # if i == 3: break

