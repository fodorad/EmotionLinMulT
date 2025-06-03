import torch
import webdataset as wds
import numpy as np
from pathlib import Path
from tqdm import tqdm


DB_PROCESSED = Path("data/db_processed/RAVDESS")


def _process_sample(sample):

    for key in ["emotion_class.npy", "emotion_intensity.npy", "clip.npy", "wavlm_baseplus.npy"]:
        if key in sample:
            sample[key] = torch.from_numpy(sample[key].copy())

    # handle masks
    mask_keys = [key for key in sample.keys() if key.endswith("_mask.npy")]
    for key in mask_keys:
        sample[key] = torch.from_numpy(sample[key].copy()).bool()

    return sample


def get_shard_urls(subset: str):
    webdataset_dir = DB_PROCESSED / "webdataset"
    pattern = f"ravdess_{subset}_*.tar"
    shard_files = sorted(webdataset_dir.glob(pattern))
    urls = [str(f) for f in shard_files]
    if not urls: raise FileNotFoundError(f"No shards found for pattern {subset} in {webdataset_dir}")
    return urls


def create_dataset(subset: str):
    urls = get_shard_urls(subset)

    pipeline = [
        wds.WebDataset(urls, shardshuffle=False),
        wds.decode(),
        wds.map(_process_sample),
    ]

    if subset == "train":
        pipeline.insert(1, wds.shuffle(5000))

    return wds.DataPipeline(*pipeline)


if __name__ == "__main__":

    test_dataset = create_dataset("test")
    for i, sample in tqdm(enumerate(test_dataset), desc="Loading test samples"):
        print(f"[test] {i}")
        print(list(sample.keys()))
        if i == 10: break
