import yaml
import psutil
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import lightning.pytorch as L
from emotionlinmult.train.dataset_afewva import AfewvaDataset
from emotionlinmult.train.dataset_ravdess import RavdessDataset
from emotionlinmult.train.utils import load_config


class DatasetFactory:
    """Factory class to create datasets dynamically based on configuration."""
    
    # Mapping of dataset names to their corresponding Dataset subclasses
    DATASET_CLASSES = {
        'afewva': AfewvaDataset,
        'affwild2': ...,
        'celebvhq': ...,
        'cremad': ...,
        'mead': ...,
        'meld': ...,
        'ravdess': RavdessDataset,
    }


    @staticmethod
    def create_datasets(db_names: list[str], subset: str, config: dict) -> list[Dataset]:
        """
        Create a list of datasets based on database names.

        Args:
            db_names (list[str]): List of database names.
            subset (str): Subset of the dataset ('train', 'valid', 'test').
            shared_args (dict): Common arguments for all datasets.

        Returns:
            list[Dataset]: List of dataset objects.
        """
        datasets = []
        for db_name in db_names:
            dataset_class = DatasetFactory.DATASET_CLASSES.get(db_name)
            if dataset_class is None:
                raise ValueError(f"Unsupported dataset: {db_name}")
            
            datasets.append(dataset_class(subset=subset, config=config))
        return datasets


class UnifiedWrapper(Dataset):

    def __init__(self, dataset: Dataset, config: dict):
        self.dataset = dataset
        self.time_dim_egemaps = config.get('time_dim_egemaps', 1000)
        self.time_dim_wavlm = config.get('time_dim_wavlm', 500)
        self.time_dim_frames = config.get('time_dim_frames', 300)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "sample_id":              sample.get("sample_id",              "unknown"),
            "emotion_class":          sample.get("emotion_class",          torch.tensor(-1, dtype=torch.float32)), # ()
            "emotion_intensity":      sample.get("emotion_intensity",      torch.tensor(-1, dtype=torch.float32)), # ()
            "emotion_class_mask":     sample.get("emotion_class_mask",     torch.zeros(size=(), dtype=bool)), # ()
            "emotion_intensity_mask": sample.get("emotion_intensity_mask", torch.zeros(size=(), dtype=bool)), # ()
            "valence_arousal":        sample.get("valence_arousal",        -torch.ones(size=(self.time_dim_frames,2), dtype=torch.float32)), # (T,2)
            "valence_arousal_mask":   sample.get("valence_arousal_mask",   torch.zeros(size=(self.time_dim_frames,), dtype=bool)), # (T,)
            "egemaps_lld":            sample.get("egemaps_lld",            torch.zeros(size=(self.time_dim_egemaps,25), dtype=torch.float32)), # (T, 25)
            "egemaps_lld_mask":       sample.get("egemaps_lld_mask",       torch.zeros(size=(self.time_dim_egemaps,), dtype=bool)), # (T,)
            "wavlm_baseplus":         sample.get("wavlm_baseplus",         torch.zeros(size=(12, self.time_dim_wavlm,768), dtype=torch.float32)), # (12, T, 768)
            "wavlm_baseplus_mask":    sample.get("wavlm_baseplus_mask",    torch.zeros(size=(self.time_dim_wavlm,), dtype=bool)), # (T,)
            "opengraphau":            sample.get("opengraphau",            torch.zeros(size=(self.time_dim_frames,41), dtype=torch.float32)), # (T, 41)
            "opengraphau_mask":       sample.get("opengraphau_mask",       torch.zeros(size=(self.time_dim_frames,), dtype=bool)), # (T,)
            "fabnet":                 sample.get("fabnet",                 torch.zeros(size=(self.time_dim_frames,256), dtype=torch.float32)), # (T, 256)
            "fabnet_mask":            sample.get("fabnet_mask",            torch.zeros(size=(self.time_dim_frames,), dtype=bool)), # (T,)
            "clip":                   sample.get("clip",                   torch.zeros(size=(self.time_dim_frames,1024), dtype=torch.float32)), # (T, 1024)
            "clip_mask":              sample.get("clip_mask",              torch.zeros(size=(self.time_dim_frames,), dtype=bool)), # (T,)
        }


class EmotionDataModule(L.LightningDataModule):

    def __init__(self, config: dict):
        super().__init__()

        self.train_db: list[str] = config['train_db']
        self.valid_db: list[str] = config['valid_db']
        self.test_db: list[str] = config['test_db']
        self.features: list[str] = config.get('features', ['egemaps_lld', 'wavlm', 'opengraphau', 'fabnet', 'clip', 'xmlroberta'])
        self.batch_size: int = config.get('batch_size', 16)
        self.window_sec: float = config.get('window_sec', 10.0)
        self.seed: int = config.get('seed', 42)
        self.num_workers = config.get('num_workers', psutil.cpu_count(logical=False))

        self.train_dataset: ConcatDataset | None = None
        self.valid_dataset: ConcatDataset | None = None
        self.test_dataset: ConcatDataset | None = None

        self.config = config


    def setup(self, stage: str | None = None):
        """
        Setup datasets for different stages of the LightningDataModule.

        Args:
            stage (str | None): Current stage ('fit', 'validate', 'test', or 'predict').
        """
        if stage in ('fit', 'validate'):
            # Load train and valid datasets
            train_datasets = DatasetFactory.create_datasets(self.train_db, "train", self.config)
            train_datasets = [UnifiedWrapper(ds, self.config) for ds in train_datasets]
            self.train_dataset = ConcatDataset(train_datasets)

            valid_datasets = DatasetFactory.create_datasets(self.valid_db, "valid", self.config)
            valid_datasets = [UnifiedWrapper(ds, self.config) for ds in valid_datasets]
            self.valid_dataset = ConcatDataset(valid_datasets)

        if stage == 'test':
            # Load test dataset
            test_datasets = DatasetFactory.create_datasets(self.test_db, "test", self.config)
            test_datasets = [UnifiedWrapper(ds, self.config) for ds in test_datasets]
            self.test_dataset = ConcatDataset(test_datasets)


    def train_dataloader(self):
        """Create DataLoader for training."""
        assert self.train_dataset is not None
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True)


    def val_dataloader(self):
        """Create DataLoader for validation."""
        assert self.valid_dataset is not None
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True)


    def test_dataloader(self):
        """Create DataLoader for testing."""
        assert self.test_dataset is not None
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True)


    def setup_and_test(self):
        test_datasets = DatasetFactory.create_datasets(self.test_db, "test", self.config)
        test_datasets = [UnifiedWrapper(ds, self.config) for ds in test_datasets]
        test_dataset = ConcatDataset(test_datasets)
        return DataLoader(test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True)


    def print_stats(self):
        print('[train dataloader]\n\tnumber of batches:', len(self.train_dataloader()))
        print('[valid dataloader]\n\tnumber of batches:', len(self.val_dataloader()))
        print('[test dataloader]\n\tnumber of batches:', len(self.test_dataloader()))


if __name__ == "__main__":
    config = load_config('configs/dataloader_av.yaml')
    dm = EmotionDataModule(config)
    dm.setup("fit")
    dl = dm.train_dataloader()
    for batch in tqdm(dl, total=len(dl)):
        x = [
            batch['emotion_class'],
            batch['emotion_intensity'],
            batch['valence'],
            batch['arousal'],
            batch['egemaps_lld'],
            batch['wavlm_baseplus'],
            batch['opengraphau'],
            batch['fabnet'],
            batch['clip'],
        ]
        print(batch['sample_id'])
        print([elem.shape for elem in x])
        exit()