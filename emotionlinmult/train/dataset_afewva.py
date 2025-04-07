import json
from pathlib import Path
import numpy as np
import torch
import pickle
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from exordium.utils.padding import pad_or_crop_time_dim


DB = Path("data/db/AFEW-VA")
DB_PROCESSED = Path("data/db_processed/AFEW-VA")
SUBSET_IDS = {
    'train': [f"{i:03}" for i in range(1, 601)][0:360],
    'valid': [f"{i:03}" for i in range(1, 601)][360:480],
    'test':  [f"{i:03}" for i in range(1, 601)][480:],
}


class AfewvaDataset(Dataset):

    def __init__(self, subset: str, config: dict | None = None):
        """
        Args:
            config (dict): Configuration dictionary with:
                - db (str): Path to the dataset directory.
                - db_processed (str): Path to the processed dataset directory.
                - time_dim (int): Number of frames in each sample (window size).
        """
        if config is None:
            config = {}

        self.subset = subset
        self.subset_ids = SUBSET_IDS[subset]
        self.db = config.get("db", DB)
        self.db_processed = config.get("db_processed", DB_PROCESSED)
        self.time_dim_frames = config.get("time_dim_frames", 300)

        self.clip_dir = self.db_processed / 'clip'
        self.fabnet_dir = self.db_processed / 'fabnet'
        self.opengraphau_dir = self.db_processed / 'opengraphau'

        self.annotations = self._read_annotations()


    def _read_annotations(self):
        filepaths = sorted(list(Path(self.db).glob("**/*.json")))
        annotation_data = {}
        for filepath in filepaths:
            sample_id = filepath.stem
            with open(filepath, "r") as f:
                annotation = json.load(f)
            annotation_data[sample_id] = annotation
        return annotation_data


    def __len__(self):
        return len(self.subset_ids)


    def _load_pkl_feature(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data[1] # 0: frame_ids; 1: feature

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to fetch.
        
        Returns:
            sample (dict): A dictionary containing sample features and annotations
        """
        video_id = self.subset_ids[idx]
        video_annotation = self.annotations[video_id]
        frame_ids = list(video_annotation["frames"].keys())

        # Load annotations
        valence = torch.tensor([video_annotation["frames"][frame_id]["valence"] for frame_id in frame_ids], dtype=torch.float32) # (N,)
        arousal = torch.tensor([video_annotation["frames"][frame_id]["arousal"] for frame_id in frame_ids], dtype=torch.float32) # (N,)
        valence_arousal = torch.stack([valence, arousal], dim=1) # (N, 2)

        # Load features
        clip = torch.tensor(self._load_pkl_feature(self.clip_dir / f'{video_id}.pkl'), dtype=torch.float32) # (N, 1024)
        fabnet = torch.tensor(self._load_pkl_feature(self.fabnet_dir / f'{video_id}.pkl'), dtype=torch.float32) # (N, 256)
        opengraphau = torch.tensor(self._load_pkl_feature(self.opengraphau_dir / f'{video_id}.pkl'), dtype=torch.float32) # (N, 41)

        assert all(x.shape[0] == clip.shape[0] for x in [fabnet, opengraphau, valence, arousal])

        # padding to time_dim
        valence_arousal, mask = pad_or_crop_time_dim(valence_arousal, self.time_dim_frames) # (T,) and (T,)
        clip, _ = pad_or_crop_time_dim(clip, self.time_dim_frames) # (T, 1024)
        fabnet, _ = pad_or_crop_time_dim(fabnet, self.time_dim_frames) # (T, 256)
        opengraphau, _ = pad_or_crop_time_dim(opengraphau, self.time_dim_frames) # (T, 41)

        return {
            "sample_id": f"afewva_{video_id}",
            "valence_arousal": valence_arousal / 10., # (T, 2); [-10..10] -> [-1..1]
            "valence_arousal_mask": mask, # (T,)
            "clip": clip, # (T, 1024)
            "clip_mask": mask, # (T,)
            "fabnet": fabnet, # (T, 256)
            "fabnet_mask": mask, # (T,)
            "opengraphau": opengraphau, # (T, 41)
            "opengraphau_mask": mask, # (T,)
        }


if __name__ == "__main__":
    ds = AfewvaDataset(subset='test')

    for i in tqdm(range(len(ds)), total=len(ds)):
        sample = ds[i]
        print('sample id:', sample["sample_id"])
        print('valence_arousal shape:', sample["valence_arousal"].shape)
        print('opengraphau shape:', sample["opengraphau"].shape)
        print('fabnet shape:', sample["fabnet"].shape)
        print('clip shape:', sample["clip"].shape)
        break

    dl = DataLoader(ds, batch_size=8)
    for i in tqdm(dl, total=len(dl)):
        pass