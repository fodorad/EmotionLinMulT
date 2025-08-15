import json
from pathlib import Path
from tqdm import tqdm
import pprint
import numpy as np
import torch
import webdataset as wds
from emotionlinmult.preprocess.MOSEI import DB_PROCESSED


def _process_sample(sample):

    for key in ["sentiment.npy", "clip.npy", "wavlm_baseplus.npy", "xml_roberta.npy"]:
        if key in sample:
            sample[key] = torch.from_numpy(sample[key])

    # handle masks
    mask_keys = [key for key in sample.keys() if key.endswith("_mask.npy")]
    for key in mask_keys:
        sample[key] = torch.from_numpy(sample[key]).bool()

    sample['dataset'] = 'MOSEI'
    sample['sentiment_mask.npy'] = torch.tensor(True, dtype=torch.bool)
    return sample


def get_shard_urls(subset: str):
    webdataset_dir = DB_PROCESSED / "webdataset"
    pattern = f"mosei_{subset}_*.tar"
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


def count_samples(cache_path: Path = DB_PROCESSED / "count_samples.json", verbose: bool = False) -> dict[str, int]:
    if cache_path.exists():
        with open(cache_path, "r") as f:
            d = json.load(f)
    else:
        d = {subset: sum(1 for _ in create_dataset(subset)) for subset in ["train", "valid", "test"]}
        with open(cache_path, "w") as f:
            json.dump(d, f)
    if verbose: pprint.pprint(d)
    return d


def calculate_sentiment_distribtuion():
    total = 0
    sentiment_dist = {}
    for subset in ['train', 'valid', 'test']:
        dataset = create_dataset(subset)

        sentiment = []
        subset_total = 0
        for i, sample in tqdm(enumerate(dataset), desc=f"Loading {subset} samples"):
            sentiment.append(sample['sentiment.npy'].item())
            subset_total += 1
        total += subset_total
        sentiment_dist[subset] = {
            'sentiment': sentiment,
            'total': subset_total
        }

        print(f'[{subset}] Sentiment:')
        pprint.pprint(sentiment)
        print(f'[{subset}] Subset total:', subset_total)
        print("="*20)
    print('Total:', total)

    output_file = DB_PROCESSED / f"sentiment_distribution.json"
    with open(output_file, "w") as f:
        json.dump(sentiment_dist, f)
    print("Sentiment distribution saved to", output_file)


if __name__ == "__main__":

    count_samples()
    calculate_sentiment_distribtuion()
    exit()

    test_dataset = create_dataset("test")
    for i, sample in tqdm(enumerate(test_dataset), desc="Loading test samples"):
        print(f"[test] {i}")
        print(sample.keys())
        if i == 10: break
