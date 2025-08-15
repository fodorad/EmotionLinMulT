import torch
import json
import pprint
import webdataset as wds
import numpy as np
from pathlib import Path
from tqdm import tqdm
from emotionlinmult.preprocess.AFEWVA import DB_PROCESSED


def _process_sample(sample):

    for key in ["valence.npy", "arousal.npy", "clip_face.npy"]:
        if key in sample:
            sample[key] = torch.from_numpy(sample[key].copy())
    
    # Remap valence/arousal from [-10,10] to [-1,1]
    for key in ["valence.npy", "arousal.npy"]:
        sample[key] = sample[key] / 10.0

    # handle masks
    mask_keys = [key for key in sample.keys() if key.endswith("_mask.npy")]
    for key in mask_keys:
        sample[key] = torch.from_numpy(sample[key].copy()).bool()

    # rename clip_face to clip
    if 'clip_face.npy' in sample:
        sample['clip.npy'] = sample['clip_face.npy']
        del sample['clip_face.npy']
    
    if 'clip_face_mask.npy' in sample:
        sample['clip_mask.npy'] = sample['clip_face_mask.npy']
        del sample['clip_face_mask.npy']

    sample['dataset'] = 'AFEW-VA_face'
    return sample


def get_shard_urls(subset: str):
    webdataset_dir = DB_PROCESSED / "webdataset_face"
    pattern = f"afewva_{subset}_*.tar"
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


def count_samples(cache_path: Path = DB_PROCESSED / "count_samples_face.json", verbose: bool = False) -> dict[str, int]:
    if cache_path.exists():
        with open(cache_path, "r") as f:
            d = json.load(f)
    else:
        d = {subset: sum(1 for _ in create_dataset(subset)) for subset in ["train", "valid", "test"]}
        with open(cache_path, "w") as f:
            json.dump(d, f)
    if verbose: pprint.pprint(d)
    return d


if __name__ == "__main__":

    count_samples(verbose=True)
    train_dataset = create_dataset_with_size("train")
    print(len(train_dataset))
    exit()

    test_dataset = create_dataset("test")
    for i, sample in tqdm(enumerate(test_dataset), desc="Loading test samples"):
        print(f"[test] {i}")
        print(list(sample.keys()))
        if i == 10: break
