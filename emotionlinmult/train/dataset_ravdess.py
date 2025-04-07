import json
from pathlib import Path
from enum import Enum
import numpy as np
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from exordium.utils.padding import pad_or_crop_time_dim
from exordium.utils.normalize import standardization, load_params_from_json
from exordium.audio.wavlm import pad_wavlm_time_dim


DB = Path("data/db/RAVDESS")
DB_PROCESSED = Path("data/db_processed/RAVDESS")

SUBSET_IDS = {
    'train': list(map(str, range(1,25)))[0:16],
    'valid': list(map(str, range(1,25)))[16:20],
    'test':  list(map(str, range(1,25)))[20:],
}

class OriginalRavdessEmotionClass(Enum):
    NEUTRAL = 1
    CALM = 2
    HAPPY = 3
    SAD = 4
    ANGRY = 5
    FEARFUL = 6
    DISGUST = 7
    SURPRISED = 8

class OriginalRavdessEmotionIntensity(Enum):
    NORMAL = 1
    STRONG = 2

EMOTION_TO_CLASS = {e.name.lower(): e.value - 1 for e in OriginalRavdessEmotionClass}
CLASS_TO_EMOTION = {e.value - 1: e.name.lower() for e in OriginalRavdessEmotionClass}
INTENSITY_TO_CLASS = {e.name.lower(): e.value - 1 for e in OriginalRavdessEmotionIntensity}
CLASS_TO_INTENSITY = {e.value - 1: e.name.lower() for e in OriginalRavdessEmotionIntensity}


class RavdessDataset(Dataset):


    def __init__(self, subset: str, config: dict | None = None):
        if config is None: config = {}

        self.subset = subset
        self.db = config.get("db", DB)
        self.db_processed = config.get("db_processed", DB_PROCESSED)
        self.time_dim_egemaps = config.get("time_dim_egemaps", 1000)
        self.time_dim_wavlm = config.get("time_dim_wavlm", 500)
        self.time_dim_frames = config.get("time_dim_frames", 300)

        self.clip_dir = self.db_processed / 'clip'
        self.fabnet_dir = self.db_processed / 'fabnet'
        self.opengraphau_dir = self.db_processed / 'opengraphau'
        self.egemaps_lld_dir = self.db_processed / 'egemaps_lld'
        self.wavlm_baseplus_dir = self.db_processed / 'wavlm_baseplus'

        self.unique_ids = sorted(list(set(
            [elem.stem for elem in self.egemaps_lld_dir.glob('*.npy')] + 
            [elem.stem for elem in self.fabnet_dir.glob('*.pkl')]
        )))

        # Filter data
        if config.get('ravdess_speech_only', True):
            self.unique_ids = [sample_id for sample_id in self.unique_ids if sample_id.split('-')[1] == '01'] # song is skipped

        self.modality = config.get('ravdess_modality', 'av')
        if self.modality == 'a':
            self.unique_ids = [sample_id for sample_id in self.unique_ids if sample_id.split('-')[0] == '03']
        elif self.modality == 'v':
            self.unique_ids = [sample_id for sample_id in self.unique_ids if sample_id.split('-')[0] == '02']
        else: # av
            self.unique_ids = [sample_id for sample_id in self.unique_ids if sample_id.split('-')[0] == '01']

        if config.get('ravdess_split', 'random') == 'random':
            
            train_ids, temp_data = train_test_split(
                self.unique_ids, test_size=0.3, random_state=config.get('seed', 42)
            )
            valid_ids, test_ids = train_test_split(
                temp_data, test_size=0.5, random_state=config.get('seed', 42)
            )
            self.unique_ids = {
                'train': sorted(train_ids),
                'valid': sorted(valid_ids),
                'test': sorted(test_ids),
            }[subset]
        else:
            subset_ids = SUBSET_IDS[subset]
            self.unique_ids = sorted(
                [elem for elem in self.unique_ids if elem.split('-')[-1] in subset_ids]
            )

        self._load_norm_params()


    def __len__(self):
        return len(self.unique_ids)


    def _load_norm_params(self):
        opengraphau_mean, opengraphau_std = load_params_from_json('data/norm/opengraphau.json')
        egemaps_lld_mean, egemaps_lld_std = load_params_from_json('data/norm/ravdess_egemaps_lld.json')
        self.norm = {
            'opengraphau': {'mean': opengraphau_mean, 'std': opengraphau_std},
            'egemaps_lld': {'mean': egemaps_lld_mean, 'std': egemaps_lld_std},
        }


    def _load_fabnet_feature(self, path) -> torch.Tensor:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        feature = data[1] # 0: frame_ids; 1: feature
        feature = torch.tensor(feature, dtype=torch.float32) # (N, 256)
        return feature


    def _load_opengraphau_feature(self, path) -> torch.Tensor:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        feature = data[1] # 0: frame_ids; 1: feature
        feature = torch.tensor(feature, dtype=torch.float32) # (N, 41)
        feature = standardization(feature, self.norm['opengraphau']['mean'], self.norm['opengraphau']['std'])
        return feature


    def _load_wavlm_feature(self, path) -> tuple[torch.Tensor, torch.Tensor]:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        feature = torch.stack([torch.tensor(elem, dtype=torch.float32) for elem in data], dim=0) # L layers (L, N, 768)
        feature_padded, mask = pad_wavlm_time_dim(feature, self.time_dim_wavlm) # (L, T, 768)
        return feature_padded, mask


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to fetch.
        
        Returns:
            sample (dict): A dictionary containing the sample features and annotations
        """
        video_id = self.unique_ids[idx]
        annotation = self._get_annotation(video_id)

        # Load annotations
        emotion_class = torch.tensor(annotation['emotion_class'], dtype=torch.float32) # ()
        emotion_intensity = torch.tensor(annotation['emotion_intensity'], dtype=torch.float32) # ()

        # Load features
        if annotation['modality'] in [1, 2]: # AV or V
            clip = torch.tensor(np.load(self.clip_dir / f'{video_id}.npy'), dtype=torch.float32) # (N, 1024)
            fabnet = self._load_fabnet_feature(self.fabnet_dir / f'{video_id}.pkl')
            opengraphau = self._load_opengraphau_feature(self.opengraphau_dir / f'{video_id}.pkl') # (N, 41)
            clip, clip_mask = pad_or_crop_time_dim(clip, self.time_dim_frames) # (T, 1024)
            fabnet, fabnet_mask = pad_or_crop_time_dim(fabnet, self.time_dim_frames) # (T, 256)
            opengraphau, opengraphau_mask = pad_or_crop_time_dim(opengraphau, self.time_dim_frames) # (T, 41)
        else:
            clip =             torch.zeros(size=(self.time_dim_frames, 1024), dtype=torch.float32)
            fabnet =           torch.zeros(size=(self.time_dim_frames, 256), dtype=torch.float32)
            opengraphau =      torch.zeros(size=(self.time_dim_frames, 41), dtype=torch.float32)
            clip_mask =        torch.zeros(size=(self.time_dim_frames,), dtype=bool)
            fabnet_mask =      torch.zeros(size=(self.time_dim_frames,), dtype=bool)
            opengraphau_mask = torch.zeros(size=(self.time_dim_frames,), dtype=bool)

        if annotation['modality'] in [1, 3]: # AV or A
            egemaps_lld = torch.tensor(np.load(self.egemaps_lld_dir / f'{video_id}.npy'), dtype=torch.float32) # (N, 25)
            egemaps_lld, egemaps_lld_mask = pad_or_crop_time_dim(egemaps_lld, self.time_dim_egemaps) # (T, 1024)
            wavlm_baseplus, wavlm_baseplus_mask = self._load_wavlm_feature(self.wavlm_baseplus_dir / f'{video_id}.pkl') # (L, T, 768)
        else:
            egemaps_lld =         torch.zeros(size=(self.time_dim_egemaps, 25), dtype=torch.float32)
            wavlm_baseplus =      torch.zeros(size=(12, self.time_dim_wavlm, 768), dtype=torch.float32) # wavlm_baseplus has 12 layers and 768 as feature_dim
            egemaps_lld_mask =    torch.zeros(size=(self.time_dim_egemaps,), dtype=bool)
            wavlm_baseplus_mask = torch.zeros(size=(self.time_dim_wavlm,), dtype=bool)

        return {
            "sample_id": f"ravdess_{video_id}",
            "emotion_class": emotion_class, # ()
            "emotion_class_mask": emotion_class != -1, # ()
            "emotion_class_name": CLASS_TO_EMOTION[int(emotion_class)], # ()
            "emotion_intensity": emotion_intensity, # ()
            "emotion_intensity_mask": emotion_intensity != -1, # ()
            "emotion_intensity_name": CLASS_TO_INTENSITY[int(emotion_intensity)], # ()
            "clip": clip, # (T_frames, 1024)
            "clip_mask": clip_mask, # (T_frames,)
            "fabnet": fabnet, # (T_frames, 256)
            "fabnet_mask": fabnet_mask, # (T_frames,)
            "opengraphau": opengraphau, # (T_frames, 41)
            "opengraphau_mask": opengraphau_mask, # (T_frames,)
            "egemaps_lld": egemaps_lld, # (T_egemaps, 25)
            "egemaps_lld_mask": egemaps_lld_mask, # (T_egemaps,)
            "wavlm_baseplus": wavlm_baseplus, # (12, T_wavlm, 768)
            "wavlm_baseplus_mask": wavlm_baseplus_mask # (T_wavlm,)
        }


    def _get_annotation(self, video_id: str):
        """
        Parses a filename to extract identifiers based on the predefined format.

        Args:
            video_id (str): The filename to parse (e.g., '02-01-06-01-02-01-12.mp4').

        Returns:
            dict: A dictionary containing the extracted annotations.
        """
        components = video_id.split('-')
        if len(components) != 7:
            raise ValueError(f"Filename format is incorrect: {video_id}")

        annotation = {
            "modality": int(components[0]),
            "vocal_channel": int(components[1]),
            "emotion_class": int(components[2]) - 1, # map the 1-8 classes to 0-7
            "emotion_intensity": int(components[3]) - 1, # map the 1-2 classes to 0-1
            "statement": int(components[4]),
            "repetition": int(components[5]),
            "actor": int(components[6]),
        }

        return annotation


if __name__ == "__main__":
    config = {'ravdess_speech_only': True, 'ravdess_modality': 'v'}
    ds_train = RavdessDataset(subset='train', config=config)
    ds_valid = RavdessDataset(subset='valid', config=config)
    ds_test = RavdessDataset(subset='test', config=config)
    print('train subset size:', len(ds_train))
    print('valid subset size:', len(ds_valid))
    print('test subset size:', len(ds_test))
    exit()

    for i in tqdm(range(len(ds_test)), total=len(ds_test)):
        sample = ds_test[i]
        print('sample id:', sample["sample_id"])
        print('emotion_class:', sample["emotion_class"])
        print('emotion_intensity:', sample["emotion_intensity"])
        print('egemaps shape:', sample["egemaps_lld"].shape)
        print('wavlm shape:', sample["wavlm_baseplus"].shape)
        print('opengraphau shape:', sample["opengraphau"].shape)
        print('fabnet shape:', sample["fabnet"].shape)
        print('clip shape:', sample["clip"].shape)
        break

    dl_test = DataLoader(ds_test, batch_size=30)
    for batch in tqdm(dl_test, total=len(dl_test)):
        pass