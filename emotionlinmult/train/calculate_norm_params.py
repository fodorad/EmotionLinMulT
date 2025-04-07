import pickle
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from exordium.utils.normalize import get_mean_std, save_params_to_json


class TensorDataset(Dataset):

    def __init__(self, file_paths):
        self.file_paths = file_paths

        tensors = []
        for filepath in tqdm(self.file_paths):
            filepath = Path(filepath)

            if filepath.suffix == '.npy':
                np_array = np.load(filepath)
            else: # .pkl
                with open(filepath, 'rb') as f:
                    _, np_array = pickle.load(f)

            tensors.append(torch.FloatTensor(np_array))

        self.stacked_tensor = torch.cat(tensors, dim=0) # (T, F)

    def __len__(self):
        return len(self.stacked_tensor)

    def __getitem__(self, idx):
        return self.stacked_tensor[idx,:], 0


def calculate_egemaps_params():
    dirnames = [
        Path(f"data/db_processed/RAVDESS/egemaps_lld"),
    ]

    feature_paths = []
    for dirname in dirnames:
        feature_paths += list(dirname.glob('*.npy'))

    dataset = TensorDataset(feature_paths)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False) 
    mean, std = get_mean_std(dataloader, ndim=2) # (B*T, C)
    save_params_to_json(mean, std, 'data/norm/egemaps_lld.json')


def calculate_opengraphau_params():
    dirnames = [
        Path(f"data/db_processed/AFEW-VA/opengraphau"),
        Path(f"data/db_processed/RAVDESS/opengraphau"),
    ]

    feature_paths = []
    for dirname in dirnames:
        feature_paths += list(dirname.glob('*.pkl'))

    dataset = TensorDataset(feature_paths)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False)
    mean, std = get_mean_std(dataloader, ndim=2) # (B*T, C)
    save_params_to_json(mean, std, 'data/norm/opengraphau.json')


def calculate_ravdess_opengraphau_params():
    dirnames = [
        Path(f"data/db_processed/RAVDESS/opengraphau"),
    ]

    feature_paths = []
    for dirname in dirnames:
        feature_paths += list(dirname.glob('*.pkl'))

    dataset = TensorDataset(feature_paths)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False)
    mean, std = get_mean_std(dataloader, ndim=2) # (B*T, C)
    save_params_to_json(mean, std, 'data/norm/ravdess_opengraphau.json')


def calculate_ravdess_egemaps_params():
    from emotionlinmult.train.dataset_ravdess import RavdessDataset
    config = {'ravdess_speech_only': True, 'ravdess_modality': 'a'}
    unique_ids = RavdessDataset(subset='train', config=config).unique_ids
    print('number of samples:', len(unique_ids))

    feature_paths = [
        elem for elem in list(Path(f"data/db_processed/RAVDESS/egemaps_lld").glob('*.npy'))
        if elem.stem in unique_ids
    ]

    dataset = TensorDataset(feature_paths)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False) 
    mean, std = get_mean_std(dataloader, ndim=2) # (B*T, C)
    save_params_to_json(mean, std, 'data/norm/ravdess_egemaps_lld.json')


if __name__ == '__main__':
    #calculate_egemaps_params()
    #calculate_opengraphau_params()
    calculate_ravdess_egemaps_params()