import json
from pathlib import Path
import numpy as np
import torch
import pprint
from tqdm import tqdm
import webdataset as wds
from emotionlinmult.preprocess import MEAD_EMOTION_MAPPING, MEAD_INTENSITY_MAPPING
from emotionlinmult.preprocess.MEAD import DB_PROCESSED


def _process_sample(sample, drop_feature_list: list[str] | None = None):

    for key in ["camera_id.npy", "emotion_class.npy", "emotion_intensity.npy", "clip.npy", "wavlm_baseplus.npy"]:
        if key in sample:
            sample[key] = torch.from_numpy(sample[key].copy())

    # unify intensity class values: from 1 weak, 2 normal, 3 strong to 0, 1, 2
    sample['emotion_intensity.npy'] = torch.tensor(MEAD_INTENSITY_MAPPING.to_unified_id(sample['emotion_intensity.npy'].item()), dtype=torch.int64)

    # convert emotion class to unified emotion class
    sample['emotion_class.npy'] = torch.tensor(MEAD_EMOTION_MAPPING.to_unified_id(sample['emotion_class.npy'].item()), dtype=torch.int64)

    # handle masks
    mask_keys = [key for key in sample.keys() if key.endswith("_mask.npy")]
    for key in mask_keys:
        sample[key] = torch.from_numpy(sample[key].copy()).bool()

    sample['dataset'] = 'MEAD'
    sample['emotion_class_mask.npy'] = torch.tensor(True, dtype=torch.bool)
    sample['emotion_intensity_mask.npy'] = torch.tensor(True, dtype=torch.bool)

    if drop_feature_list is not None and 'clip' not in drop_feature_list:
        sample['clip.npy'] = torch.zeros_like(sample['clip.npy'], dtype=torch.float32)
        sample['clip_mask.npy'] = torch.zeros_like(sample['clip_mask.npy'], dtype=torch.bool)

    if drop_feature_list is not None and 'wavlm_baseplus' not in drop_feature_list:
        sample['wavlm_baseplus.npy'] = torch.zeros_like(sample['wavlm_baseplus.npy'], dtype=torch.float32)
        sample['wavlm_baseplus_mask.npy'] = torch.zeros_like(sample['wavlm_baseplus_mask.npy'], dtype=torch.bool)

    return sample

 
def get_shard_urls(subset: str):
    webdataset_dir = DB_PROCESSED / "webdataset"
    pattern = f"mead_{subset}_*.tar"
    shard_files = sorted(webdataset_dir.glob(pattern))
    urls = [str(f) for f in shard_files]
    if not urls: raise FileNotFoundError(f"No shards found for pattern {pattern} in {webdataset_dir}")
    return urls


def create_dataset(
        subset: str,
        camera_id: int | None = None,
        drop_feature_list: list[str] | None = None,
        shuffle_buffer_size: int = 300,
        no_contempt: bool = True, # TODO: POSTER ONLY
    ): # camera_id 1: front
    urls = get_shard_urls(subset)

    pipeline = [
        wds.WebDataset(urls, empty_check=False, shardshuffle=100 if subset == 'train' else False),
        wds.decode(),
    ]

    if camera_id is not None:
        pipeline += [
            wds.select(lambda sample: sample['camera_id.npy'].item() == camera_id)
        ]

    pipeline += [
        wds.map(lambda x: _process_sample(x, drop_feature_list=drop_feature_list)),
        wds.shuffle(shuffle_buffer_size if subset == 'train' else False),
    ]

    if no_contempt:
        pipeline += [
            wds.select(lambda sample: sample['emotion_class.npy'].item() != 7)
        ]

    return wds.DataPipeline(*pipeline)


def create_dataset_with_size(subset: str, epoch_size: int | None = None):
    wds_dataset = create_dataset(subset)
    if epoch_size is None:
        epoch_size = count_samples()[subset]
    wds_dataset = wds_dataset.with_epoch(epoch_size).with_length(epoch_size)
    return wds_dataset


def count_samples(cache_path: Path = DB_PROCESSED / "count_samples.json", verbose: bool = False) -> dict[str, int]:
    if cache_path.exists():
        with open(cache_path, "r") as f:
            d = json.load(f)
    else:
        d = {}
        for subset in ["train", "valid", "test"]:
            dataset = create_dataset(subset, camera_id=None)
            d[subset] = 0
            for ind, _ in enumerate(dataset):
                d[subset] += 1
                if ind % 1000 == 0: print(f"Counted {ind} samples for {subset}")
        with open(cache_path, "w") as f:
            json.dump(d, f)
    if verbose: pprint.pprint(d)
    return d


def calculate_class_distribution():
    """Calculate and return class distribution statistics for MEAD dataset.
    
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
        unified_intensity_counts = defaultdict(int)
        dataset_intensity_counts = defaultdict(int)
        subset_total = 0
        
        # Process samples
        for sample in tqdm(dataset, desc=f"Processing {subset} samples"):
            # Get unified class and intensity
            unified_class_id = int(sample['emotion_class.npy'])
            unified_intensity_id = int(sample['emotion_intensity.npy'])
            
            # Get unified names
            unified_emotion_name = MEAD_EMOTION_MAPPING.unified_name(unified_class_id)
            unified_intensity_name = MEAD_INTENSITY_MAPPING.unified_name(unified_intensity_id)
            
            # Get original dataset class and intensity (reverse mapping from unified to original)
            original_class_id = MEAD_EMOTION_MAPPING.to_orig_id(unified_class_id)
            original_class_name = MEAD_EMOTION_MAPPING.to_orig_name(original_class_id)
            
            # For intensity, we need to find the original ID that maps to this unified intensity
            original_intensity_id = MEAD_INTENSITY_MAPPING.to_orig_id(unified_intensity_id)
            original_intensity_name = MEAD_INTENSITY_MAPPING.to_orig_name(original_intensity_id)
            
            # Update counts
            unified_emotion_counts[unified_emotion_name] += 1
            dataset_emotion_counts[original_class_name] += 1
            unified_intensity_counts[unified_intensity_name] += 1
            dataset_intensity_counts[original_intensity_name] += 1
            subset_total += 1
        
        # Store results for this subset
        results[subset] = {
            'unified_emotion_class': dict(sorted(unified_emotion_counts.items())),
            'dataset_emotion_class': dict(sorted(dataset_emotion_counts.items())),
            'unified_intensity': dict(sorted(unified_intensity_counts.items())),
            'dataset_intensity': dict(sorted(dataset_intensity_counts.items())),
            'subset_total': subset_total
        }

    # Print summary
    for subset in ['train', 'valid', 'test']:
        print(f"\n=== {subset.upper()} ===")
        print("Unified Emotion Classes:")
        pprint.pprint(results[subset]['unified_emotion_class'])
        print("\nDataset Emotion Classes:")
        pprint.pprint(results[subset]['dataset_emotion_class'])
        print("\nUnified Intensities:")
        pprint.pprint(results[subset]['unified_intensity'])
        print("\nDataset Intensities:")
        pprint.pprint(results[subset]['dataset_intensity'])
        print(f"\nSubset total: {results[subset]['subset_total']}")
        print("=" * 50)

    # Save to file
    output_file = DB_PROCESSED / "class_distribution.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nClass distribution saved to {output_file}")

    return results


if __name__ == "__main__":

    count_samples(verbose=True)
    #calculate_class_distribution()
    #exit()

    ds = create_dataset("test")
    for i, sample in tqdm(enumerate(ds), desc="Loading test samples"):
        print(f"[test] sample {i}")
        #if i == 10: break

    """
    for camera_id in range(0, 7):
        test_dataset = create_dataset("test", camera_id=camera_id)
        for i, sample in tqdm(enumerate(test_dataset), desc="Loading test samples"):
            print(f"\n[test] {i}: {sample['__key__']} {sample['camera_id.npy'].item()} {sample['emotion_class.npy'].item()} {sample['emotion_intensity.npy'].item()}")
            #if i == 3: 
            break
    """
