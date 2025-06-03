import webdataset as wds
import torch
import numpy as np
from pathlib import Path
import lightning.pytorch as L
from functools import partial
from emotionlinmult.preprocess.MELD.dataset import create_dataset as create_meld_dataset
from emotionlinmult.preprocess.MOSEI.dataset import create_dataset as create_mosei_dataset
from emotionlinmult.preprocess.AFEW_VA.dataset import create_dataset as create_afewva_dataset
from emotionlinmult.preprocess.AffWild2.dataset_expr import create_dataset as create_affwild2_expr_dataset
from emotionlinmult.preprocess.AffWild2.dataset_va import create_dataset as create_affwild2_va_dataset
from emotionlinmult.preprocess.CREMA_D.dataset import create_dataset as create_cremad_dataset
from emotionlinmult.preprocess.MEAD.dataset import create_dataset as create_mead_dataset
from emotionlinmult.preprocess.RAVDESS.dataset import create_dataset as create_ravdess_dataset
from emotionlinmult.preprocess.CelebV_HQ.dataset import create_dataset as create_celebv_hq_dataset

"""
def create_multi_loader():
    # Process each dataset separately before combining
    pipeline = [
        wds.Chain(
            wds.WebDataset(dataset1_urls).map(dataset1_transform),
            wds.WebDataset(dataset2_urls).map(dataset2_transform)
        ),
        wds.shuffle(1000),
        wds.batched(batch_size)
    ]
    return wds.WebLoader(wds.DataPipeline(*pipeline))
"""


class DatasetFactory:
    """Factory class to create datasets dynamically based on configuration."""

    DATASET_CLASSES = {
        'meld': create_meld_dataset,
        'mosei': create_mosei_dataset,
        'afewva': create_afewva_dataset,
        'affwild2_expr': create_affwild2_expr_dataset,
        'affwild2_va': create_affwild2_va_dataset,
        'cremad': create_cremad_dataset,
        'mead': create_mead_dataset,
        'ravdess': create_ravdess_dataset,
        'celebv_hq': create_celebv_hq_dataset,
    }

    @staticmethod
    def create_dataset(db_name: str, subset: str) -> wds.WebDataset:
        """
        Create a dataset based on database name.

        Args:
            db_name (str): Name of the database.
            subset (str): Subset of the dataset ('train', 'valid', 'test').

        Returns:
            wds.WebDataset: Dataset object.
        """
        dataset_class = DatasetFactory.DATASET_CLASSES.get(db_name)
        if dataset_class is None:
            raise ValueError(f"Unsupported dataset: {db_name}")
        return dataset_class(subset=subset)

    @staticmethod
    def create_datasets(db_names: list[str], subset: str) -> list[wds.WebDataset]:
        """
        Create a list of datasets based on database names.

        Args:
            db_names (list[str]): List of database names.
            subset (str): Subset of the dataset ('train', 'valid', 'test').

        Returns:
            list[wds.WebDataset]: List of dataset objects.
        """
        datasets = []
        for db_name in db_names:
            dataset = DatasetFactory.create_dataset(db_name, subset)
            datasets.append(dataset)
        return datasets

    @staticmethod
    def add_missing_features(config: dict):
        """Create zero-filled features and masks for missing data based on config"""
        time_dims = {
            'wavlm_baseplus': config.get('time_dim_wavlm', 500),
            'clip': config.get('time_dim_frames', 300),
            'xlm_roberta': config.get('time_dim_text', 120)
        }

        feature_dims = {
            'wavlm_baseplus': 768,
            'clip': 1024,
            'xlm_roberta': 768
        }

        feature_list = config.get('feature_list', ['wavlm_baseplus', 'clip', 'xlm_roberta'])
        target_list = config.get('target_list', ['sentiment', 'sentiment_class'])
        mask_list = [f'{feature}_mask' for feature in feature_list] + [f'{target}_mask' for target in target_list]

        def _add_features(sample):
            # First convert numpy arrays to tensors and remove .npy suffix
            new_sample = {}
            for key, value in sample.items():
                new_key = key.replace('.npy', '')
                if new_key not in feature_list + target_list + mask_list and new_key not in ['dataset']:
                    continue # filter non-relevant keys
                new_sample[new_key] = value
            sample = new_sample

            # Add missing features based on config
            for feature in feature_list:

                if feature not in sample:
                    if feature in time_dims:  # Modality features
                        T = time_dims[feature]
                        D = feature_dims[feature]
                        sample[feature] = torch.zeros((T, D), dtype=torch.float32)
                        sample[f'{feature}_mask'] = torch.zeros(T, dtype=torch.bool)

            for target in target_list:
                if target not in sample:
                    if target == 'sentiment':
                        sample[target] = torch.tensor(-5, dtype=torch.float32)  # -5 indicates no label
                        sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                    if target == 'sentiment_class':
                        sample[target] = torch.tensor(-1, dtype=torch.int64)  # -1 indicates no label
                        sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                    #elif target in ['valence', 'arousal']:
                    #    T = time_dims['clip']  # These are frame-wise
                    #    sample[target] = torch.full((T,), -5, dtype=torch.float32) # -5 indicates no label
                    #    sample[f'{target}_mask'] = torch.zeros(T, dtype=torch.bool) 

            return sample

        return _add_features


class MultiDatasetModule(L.LightningDataModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 8)
        self.dataset_config = config['datasets']

    def _build_dataset(self, dataset_names: list[str], subset: str, shuffle: bool = False) -> wds.WebDataset:
        """Build a unified dataset from multiple datasets"""
        # Create individual datasets and extend with missing features

         # Get all dataset pipelines
        datasets = DatasetFactory.create_datasets(dataset_names, subset)
        
        # Create a unified pipeline using Roundrobin
        union = wds.DataPipeline(
            # Combine datasets using Roundrobin, continue until longest dataset is exhausted
            wds.RoundRobin(datasets, longest=True),
            # Add missing features to samples from all datasets
            wds.map(DatasetFactory.add_missing_features(self.config)),
            # Shuffle if needed
            wds.shuffle(5000 if shuffle else False),
        )

        return union

    def setup(self, stage: str | None = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self._build_dataset(self.dataset_config['train'], subset='train', shuffle=True)
            self.val_dataset = self._build_dataset(self.dataset_config['valid'], subset='valid')
            
        if stage == 'test' or stage is None:
            self.test_dataset = self._build_dataset(self.dataset_config['test'], subset='test')

    def train_dataloader(self):
        return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return wds.WebLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":

    config = {
        'feature_list': ['wavlm_baseplus', 'clip', 'xlm_roberta'],
        'target_list': ['sentiment', 'sentiment_class'],
        'datasets': {
            'train': ['meld', 'mosei'],
            'valid': ['meld', 'mosei'],
            'test': ['meld', 'mosei']
        }
    }

    datamodule = MultiDatasetModule(config)
    datamodule.setup()

    dataloader = datamodule.test_dataloader()
    for i, batch in enumerate(dataloader):
        print(f"[Batch {i}] Keys: {list(batch.keys())}")