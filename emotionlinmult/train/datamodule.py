import webdataset as wds
import torch
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path
from time import time
import lightning.pytorch as L
from functools import partial
# AFEW_VA
from emotionlinmult.preprocess.AFEWVA.dataset_face import create_dataset as create_dataset_afewva
from emotionlinmult.preprocess.AFEWVA.dataset_face import count_samples as count_samples_afewva
# AffWild2
from emotionlinmult.preprocess.AffWild2.dataset_expr import create_dataset as create_dataset_affwild2_expr
from emotionlinmult.preprocess.AffWild2.dataset_expr import count_samples as count_samples_affwild2_expr
from emotionlinmult.preprocess.AffWild2.dataset_va import create_dataset as create_dataset_affwild2_va
from emotionlinmult.preprocess.AffWild2.dataset_va import count_samples as count_samples_affwild2_va
# CelebV_HQ
from emotionlinmult.preprocess.CelebVHQ.dataset import create_dataset as create_dataset_celebvhq
from emotionlinmult.preprocess.CelebVHQ.dataset import count_samples as count_samples_celebvhq
# CREMA-D
from emotionlinmult.preprocess.CREMAD.dataset_expr import create_dataset as create_dataset_cremad_expr
from emotionlinmult.preprocess.CREMAD.dataset_expr import count_samples as count_samples_cremad_expr
from emotionlinmult.preprocess.CREMAD.dataset_int import create_dataset as create_dataset_cremad_int
from emotionlinmult.preprocess.CREMAD.dataset_int import count_samples as count_samples_cremad_int
# MEAD
from emotionlinmult.preprocess.MEAD.dataset import create_dataset as create_dataset_mead
from emotionlinmult.preprocess.MEAD.dataset import count_samples as count_samples_mead
# MELD
from emotionlinmult.preprocess.MELD.dataset import create_dataset as create_dataset_meld
from emotionlinmult.preprocess.MELD.dataset import count_samples as count_samples_meld
# MOSEI
from emotionlinmult.preprocess.MOSEI.dataset import create_dataset as create_dataset_mosei
from emotionlinmult.preprocess.MOSEI.dataset import count_samples as count_samples_mosei
# RAVDESS
from emotionlinmult.preprocess.RAVDESS.dataset import create_dataset as create_dataset_ravdess
from emotionlinmult.preprocess.RAVDESS.dataset import count_samples as count_samples_ravdess


class DatasetFactory:
    """Factory class to create datasets dynamically based on configuration."""

    DATASET_CLASSES = {
        'afewva': (create_dataset_afewva, count_samples_afewva()),
        'affwild2_expr': (create_dataset_affwild2_expr, count_samples_affwild2_expr()),
        'affwild2_va': (create_dataset_affwild2_va, count_samples_affwild2_va()),
        'celebvhq': (create_dataset_celebvhq, count_samples_celebvhq()),
        'cremad_expr': (create_dataset_cremad_expr, count_samples_cremad_expr()),
        'cremad_int': (create_dataset_cremad_int, count_samples_cremad_int()),
        'mead': (create_dataset_mead, count_samples_mead()),
        'meld': (create_dataset_meld, count_samples_meld()),
        'mosei': (create_dataset_mosei, count_samples_mosei()),
        'ravdess': (create_dataset_ravdess, count_samples_ravdess()),
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
        dataset_class, _ = DatasetFactory.DATASET_CLASSES.get(db_name)
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
    
    def get_dataset_sizes() -> dict[str, dict[str, int]]:
        return {
            db_name: {subset: size for subset, size in DatasetFactory.DATASET_CLASSES[db_name][1].items()} 
            for db_name in DatasetFactory.DATASET_CLASSES.keys()
        }

    @staticmethod
    def add_missing_features(config: dict, subset: str):
        """Create zero-filled features and masks for missing data based on config"""
        time_dims = {
            'wavlm_baseplus': config.get('time_dim_wavlm', 500),
            'clip': config.get('time_dim_frames', 300),
            'xml_roberta': config.get('time_dim_text', 120)
        }

        feature_dims = {
            'wavlm_baseplus': 768,
            'clip': 1024,
            'xml_roberta': 768
        }

        feature_list = config.get('feature_list', ['wavlm_baseplus', 'clip', 'xml_roberta'])
        target_list = config.get('target_list', ['sentiment', 'emotion_class'])
        mask_list = [f'{feature}_mask' for feature in feature_list] + [f'{target}_mask' for target in target_list]


        def _get_available_regions(mask, min_region_size=5):
            """Convert padding mask to list of available regions"""
            if mask is None:
                return [(0, len(mask))]

            regions = []
            start = None
            for i, valid in enumerate(mask):
                if valid and start is None:
                    start = i
                elif not valid and start is not None:
                    if i - start >= min_region_size:
                        regions.append((start, i))
                    start = None
            if start is not None and len(mask) - start >= min_region_size:
                regions.append((start, len(mask)))
            return regions

        def _get_non_overlapping_blocks(regions, block_len, num_blocks, existing_blocks, min_gap=5):
            """Get non-overlapping blocks from available regions"""
            blocks = []
            for _ in range(num_blocks):
                # Flatten all regions and remove areas too close to existing blocks
                available = []
                for r_start, r_end in regions:
                    # Exclude areas too close to existing blocks
                    for block_start, block_end in blocks + existing_blocks:
                        r_start = max(r_start, block_end + min_gap)
                        if r_start >= r_end:
                            break
                    if r_end - r_start >= block_len:
                        available.append((r_start, r_end))
                
                if not available:
                    break
                    
                # Select a random region
                region_idx = torch.randint(0, len(available), (1,)).item()
                region_start, region_end = available[region_idx]
                
                # Select a random block within the region
                max_start = region_end - block_len
                if max_start <= region_start:
                    start = region_start
                else:
                    start = torch.randint(region_start, max_start, (1,)).item()
                end = start + block_len
                
                blocks.append((start, end))
            
            return blocks

        def _get_text_mask_indices(text_mask, mask_ratio=0.15):
            """Generate random mask indices for text features, considering only valid tokens"""
            # Get indices of valid tokens (where mask is True)
            valid_indices = torch.nonzero(text_mask, as_tuple=True)[0].tolist()
            if not valid_indices:
                return []
            
            # Calculate number of tokens to mask
            num_masks = max(1, int(len(valid_indices) * mask_ratio))
            
            # Randomly select tokens to mask
            selected = torch.randperm(len(valid_indices))[:num_masks].tolist()
            return [valid_indices[i] for i in selected]

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

                if config.get('keep_original', False):
                    sample[f'original_{feature}'] = sample[feature].clone()
                    sample[f'original_{feature}_mask'] = sample[f'{feature}_mask'].clone()

            for target in target_list:
                if target not in sample:
                    if target == 'sentiment':
                        sample[target] = torch.tensor(-5, dtype=torch.float32)  # -5 indicates no label
                        sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                    if target == 'sentiment_class':
                        sample[target] = torch.tensor(-1, dtype=torch.int64)  # -1 indicates no label
                        sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                    if target in ['valence', 'arousal']:
                        T = time_dims['clip']  # These are frame-wise
                        sample[target] = torch.full((T,), -5, dtype=torch.float32) # -5 indicates no label
                        sample[f'{target}_mask'] = torch.zeros(T, dtype=torch.bool)
                    if target == 'emotion_intensity':
                        sample[target] = torch.tensor(-1, dtype=torch.int64)  # -1 indicates no label
                        sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                    if target == 'emotion_class':
                        sample[target] = torch.tensor(-1, dtype=torch.int64)  # -1 indicates no label
                        sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                    if target == 'emotion_class_fw':
                        T = time_dims['clip']  # These are frame-wise
                        sample[target] = torch.full((T,), -1, dtype=torch.int64) # -1 indicates no label
                        sample[f'{target}_mask'] = torch.zeros(T, dtype=torch.bool)
                    if 'tmm_' in target:
                        feature = target.replace('tmm_', '')
                        sample[target] = sample[feature].clone()

            # Apply temporal block masking to features
            if config.get('block_dropout', 0) > 0 and \
                (subset == 'train' or (subset == 'valid' and any(['tmm_' in elem for elem in target]))):

                # Visual masking
                if 'clip' in feature_list:
                    if sample['clip_mask'].any():
                        visual_regions = _get_available_regions(sample['clip_mask'])
                        
                        visual_blocks = _get_non_overlapping_blocks(
                            visual_regions,
                            block_len=config.get('block_length', 15),
                            num_blocks=config.get('num_block_drops', 2),
                            existing_blocks=[],
                            min_gap=config.get('min_gap_between_blocks', 5)
                        )

                        visual_temporal_mask = torch.zeros_like(sample['clip_mask'])
                        for s, e in visual_blocks:
                            sample['clip'][s:e] = 0
                            visual_temporal_mask[s:e] = True
                        sample['tmm_clip_mask'] = visual_temporal_mask
                    else:
                        sample['tmm_clip_mask'] = torch.zeros_like(sample['clip_mask'], dtype=torch.bool)
                        visual_blocks = []

                # Audio masking (avoid visual mask regions)
                if 'wavlm_baseplus' in feature_list:
                    if sample['wavlm_baseplus_mask'].any():
                        audio_regions = _get_available_regions(sample['wavlm_baseplus_mask'])

                        visual_blocks_audio = []
                        for vs, ve in visual_blocks:
                            as_start = int((vs / time_dims['clip']) * time_dims['wavlm_baseplus'])
                            as_end = int((ve / time_dims['clip']) * time_dims['wavlm_baseplus'])
                            visual_blocks_audio.append((as_start, as_end))

                        audio_blocks = _get_non_overlapping_blocks(
                            audio_regions,
                            block_len=int(config.get('block_mask_length', 15) * (time_dims['wavlm_baseplus'] / time_dims['clip'])),
                            num_blocks=config.get('num_mask_blocks', 2),
                            existing_blocks=visual_blocks_audio,
                            min_gap=int(config.get('min_gap_between_blocks', 5) * (time_dims['wavlm_baseplus'] / time_dims['clip']))
                        )

                        audio_temporal_mask = torch.zeros_like(sample['wavlm_baseplus_mask'])
                        for s, e in audio_blocks:
                            sample['wavlm_baseplus'][s:e] = 0
                            audio_temporal_mask[s:e] = True
                        sample['tmm_wavlm_baseplus_mask'] = audio_temporal_mask
                    else:
                        sample['tmm_wavlm_baseplus_mask'] = torch.zeros_like(sample['wavlm_baseplus_mask'], dtype=torch.bool)

                # Text masking (random tokens)
                if 'xml_roberta' in feature_list:
                    if sample['xml_roberta_mask'].any():
                        text_temporal_mask = torch.zeros_like(sample['xml_roberta_mask'], dtype=torch.bool)
                        
                        # Get indices of tokens to mask (only from valid tokens)
                        mask_indices = _get_text_mask_indices(sample['xml_roberta_mask'])
                        
                        # Apply masking
                        if mask_indices:
                            mask_indices = torch.tensor(mask_indices, dtype=torch.long)
                            sample['xml_roberta'][mask_indices] = 0
                            text_temporal_mask[mask_indices] = True
                        
                        sample['tmm_xml_roberta_mask'] = text_temporal_mask
                    else:
                        sample['tmm_xml_roberta_mask'] = torch.zeros_like(sample['xml_roberta_mask'], dtype=torch.bool)

            # Apply feature corruption with Gaussian noise
            if config.get('feature_corruption', 0) > 0 and subset == 'train':
                for feature in feature_list:
                    if feature in sample and torch.rand(1).item() < config['feature_corruption']:
                        # Add Gaussian noise with mean=0 and std=0.1
                        noise = torch.randn_like(sample[feature]) * 0.1
                        sample[feature] = sample[feature] + noise

            # Apply modality dropout
            if config.get('modality_dropout', 0) > 0 and subset == 'train':
                available_modalities = [f for f in feature_list if f in sample and f + '_mask' in sample]
                if len(available_modalities) > 1:  # Need at least 2 modalities
                    # Decide which modalities to drop
                    to_drop = [mod for mod in available_modalities 
                             if torch.rand(1).item() < config['modality_dropout']]
                    
                    # If all would be dropped, keep one random modality
                    if len(to_drop) == len(available_modalities):
                        to_drop.remove(np.random.choice(to_drop))
                        
                    # Apply dropout to selected modalities
                    for mod in to_drop:
                        sample[mod].fill_(0)
                        sample[f'{mod}_mask'].fill_(False)

            return sample

        return _add_features


class MultiDatasetModule(L.LightningDataModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.proportion_sampling = config.get('proportion_sampling', True)
        self.batch_size = config.get('batch_size', 32)
        self.shuffle_buffer_size = config.get('shuffle_buffer_size', min(self.batch_size*5, 300))
        self.num_workers = config.get('num_workers', 8)
        self.dataset_config = config['datasets']
        self.strategy = config.get('strategy', 'auto')
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_size = None
        self.valid_size = None
        self.test_size = None

    def _build_dataset(self, dataset_names: list[str], subset: str) -> tuple[wds.WebDataset, int]:
        """Build a unified dataset from multiple datasets
        
        Returns:
            tuple: (dataset, total_samples) where total_samples is the sum of all dataset sizes
        """
        datasets = DatasetFactory.create_datasets(dataset_names, subset)
        dataset_sizes_all = DatasetFactory.get_dataset_sizes()
        dataset_sizes = {db_name: dataset_sizes_all[db_name][subset] for db_name in dataset_names}

        datasets_with_epoch = [
            ds.with_epoch(epoch_size).with_length(epoch_size)
            for ds, epoch_size in zip(datasets, dataset_sizes.values())
        ]

        total_samples = sum(dataset_sizes.values())

        print(f"\n[{subset}] Total samples: {total_samples}")
        print(f"[{subset}] Epoch sizes: {list(zip(dataset_names, dataset_sizes.values()))}\n")

        # Create a unified pipeline using Roundrobin
        union = wds.DataPipeline(
            # Combine datasets using Roundrobin, continue until longest dataset is exhausted
            wds.RoundRobin(datasets_with_epoch, longest=True),
            # Add missing features to samples from all datasets and apply modality dropout, feature corruption, block dropout if specified
            wds.map(DatasetFactory.add_missing_features(self.config, subset)),
            wds.shuffle(self.shuffle_buffer_size if subset=='train' else False),
        )
        
        # Set epoch and length for proper iteration
        union = union.with_epoch(total_samples).with_length(total_samples)
        
        return union, total_samples


    def _build_balanced_dataset(self, dataset_names: list[str], subset: str) -> tuple[wds.WebDataset, int]:
        """Build a unified proportion-sampled dataset with roundrobin from multiple datasets
        
        Returns:
            tuple: (dataset, total_samples) where total_samples is the sum of all dataset sizes
        """
        datasets = DatasetFactory.create_datasets(dataset_names, subset)
        dataset_names = self.dataset_config['train']
        dataset_sizes_all = DatasetFactory.get_dataset_sizes()
        # personal decision to limit MEAD sample size / epoch
        dataset_sizes_all['mead']['train'] = dataset_sizes_all['mead']['train'] // 10
        dataset_sizes_all['mead']['valid'] = dataset_sizes_all['mead']['valid'] // 10
        dataset_sizes = {db_name: dataset_sizes_all[db_name][subset] for db_name in dataset_names}
        total_samples = sum(dataset_sizes.values())

        # Smooth sizes by sqrt to reduce dominance of very large datasets
        smooth_sizes = [math.sqrt(s) for s in dataset_sizes.values()]
        smooth_sum = sum(smooth_sizes)
        proportions = [s / smooth_sum for s in smooth_sizes]


        # Define total epoch size (samples per epoch over combined dataset)
        total_epoch_size = total_samples # 72000
        
        # Calculate epoch size per dataset based on proportions
        epoch_sizes = [int(total_epoch_size * p) for p in proportions]

        print(f"\n[{subset}] Total samples: {total_samples}")
        print(f"[{subset}] Proportions: {list(zip(dataset_names, [round(p, 3) for p in proportions]))}")
        print(f"[{subset}] Epoch sizes: {list(zip(dataset_names, epoch_sizes))}\n")

        # Set with_epoch(epoch_size) for each dataset individually
        datasets_with_epoch = [
            ds.with_epoch(epoch_size).with_length(epoch_size)
            for ds, epoch_size in zip(datasets, epoch_sizes)
        ]

        # Create a unified pipeline using RandomMix
        union = wds.DataPipeline(
            # use randommix with weights instead of roundrobin for proportion based sampling
            wds.RandomMix(datasets_with_epoch, probs=proportions),
            # Add missing features to samples from all datasets and apply modality dropout, feature corruption, block dropout if specified
            wds.map(DatasetFactory.add_missing_features(self.config, subset)),
            wds.shuffle(self.shuffle_buffer_size if subset=='train' else False),
        )
        
        # Set epoch and length for proper iteration
        union = union.with_epoch(total_epoch_size).with_length(total_epoch_size)
        
        return union, total_epoch_size


    def wrap_dataset(self, dataset: wds.DataPipeline, subset: str) -> wds.WebLoader:

        pipeline = [
            dataset,
            wds.map(DatasetFactory.add_missing_features(self.config, subset)),
            wds.shuffle(self.shuffle_buffer_size if subset=='train' else False),
        ]

        pipeline = wds.DataPipeline(*pipeline)

        return wds.WebLoader(
            pipeline,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )


    def setup(self, stage: str | None = None):
        if self.proportion_sampling:
            buid_dataset = self._build_balanced_dataset
        else:
            buid_dataset = self._build_dataset

        if stage == 'fit' or stage is None:
            self.train_dataset, self.train_size = buid_dataset(
                self.dataset_config['train'], 
                subset='train'
            )
            self.valid_dataset, self.valid_size = buid_dataset(
                self.dataset_config['valid'], 
                subset='valid'
            )

        if stage == 'test' or stage is None: # roundrobin, not proportion based
            self.test_dataset, self.test_size = self._build_dataset(
                self.dataset_config['test'], 
                subset='test'
            )

    def train_dataloader(self):
        loader = wds.WebLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False # dataset is already shuffled in _build_dataset
        )
        total_batches = (self.train_size + self.batch_size - 1) // self.batch_size
        return loader.with_length(total_batches)

    def val_dataloader(self):
        loader = wds.WebLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )
        total_batches = (self.valid_size + self.batch_size - 1) // self.batch_size
        return loader.with_length(total_batches)

    def test_dataloader(self):
        loader = wds.WebLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )
        total_batches = (self.test_size + self.batch_size - 1) // self.batch_size
        return loader.with_length(total_batches)


    def get_class_distribution(self, subset: str):
        if subset == 'train':
            dataloader = self.train_dataloader()
        elif subset == 'valid':
            dataloader = self.val_dataloader()
        elif subset == 'test':
            dataloader = self.test_dataloader()
        else:
            raise ValueError(f"Unknown subset: {subset}")

        emotion_class_counts = torch.zeros(8, dtype=torch.int64)  # 8 unified emotion classes
        intensity_class_counts = torch.zeros(3, dtype=torch.int64)  # 3 unified intensity classes
        for batch in tqdm(dataloader, desc=f"Counting classes"):
            for elem in batch['emotion_class'][batch['emotion_class_mask']]:
                emotion_class_counts[int(elem)] += 1
            for elem in batch['emotion_intensity'][batch['emotion_intensity_mask']]:
                intensity_class_counts[int(elem)] += 1

        return emotion_class_counts, intensity_class_counts

"""
    def compute_class_weights(self):
        train_counts = self.get_class_distribution('train')
        emotion_class_weights = torch.zeros(8, dtype=torch.float32)
        intensity_class_weights = torch.zeros(3, dtype=torch.float32)
        # Avoid division by zero for weights
        emotion_class_weights = 1.0 / (train_counts[0] + 1e-6)
        intensity_class_weights = 1.0 / (train_counts[1] + 1e-6)
        # Normalize to mean=1 (over seen classes)
        emotion_class_weights = emotion_class_weights / emotion_class_weights.sum() * 8
        intensity_class_weights = intensity_class_weights / intensity_class_weights.sum() * 3
        return train_counts, emotion_class_weights, intensity_class_weights
"""

if __name__ == "__main__":

    config = {
        'feature_list': ['wavlm_baseplus', 'clip'], 
        'target_list': ['emotion_class', 'emotion_intensity'], 
        'datasets': {
            'train': ['ravdess', 'mosei', 'cremad_expr'], #, 'celebv_hq', 'cremad', 'meld', 'mosei', 'mead'],
            'valid': ['ravdess', 'mosei'],
            'test': ['ravdess']
        },
        'proportion_sampling': True,
    }

    datamodule = MultiDatasetModule(config)
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    for i, batch in enumerate(train_dataloader):
        print(f"[Batch {i}]" +
        f" ravdess: {sum([elem == 'RAVDESS' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" mosei: {sum([elem == 'MOSEI' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" cremad_expr: {sum([elem == 'CREMA-D_expr' for elem in batch['dataset']]) / len(batch['dataset'])}")
        pass
        #if i == 10: break
    exit()

    emotion_class_counts, intensity_class_counts = datamodule.get_class_distribution('train')
    print("[emotion_class]: Train counts:", emotion_class_counts)
    print("[emotion_intensity]: Train counts:", intensity_class_counts)

    exit()

    config = {
        'feature_list': ['wavlm_baseplus', 'clip', 'xml_roberta'], 
        'target_list': ['sentiment', 'tmm_wavlm_baseplus', 'tmm_clip', 'tmm_xml_roberta'], 
        'datasets': {
            'train': ['mosei'],
            'valid': ['mosei'],
            'test': ['mosei']
        },
        'keep_original': True,        # original features saved for temporal masked modeling pretraining task for reconstruction
        'modality_dropout': 0.3,      # Probability to drop modalities
        'feature_corruption': 0.1,    # Probability to corrupt features with Gaussian noise
        'block_dropout': 0.5,         # Probability to drop blocks of features
        'block_length': 15,           # ~0.5s at 30fps
        'num_block_drops': 2,         # Number of blocks to drop per sequence
        'min_gap_between_blocks': 5,  # Minimum frames between dropped blocks
    }

    datamodule = MultiDatasetModule(config)
    datamodule.setup()

    dataloader = datamodule.test_dataloader()
    for i, batch in enumerate(dataloader):
        print(f"[Batch {i}] Keys: {list(batch.keys())}")
        #if i == 10: break
