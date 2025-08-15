import webdataset as wds
import torch
import numpy as np
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from time import time
import lightning.pytorch as L
from emotionlinmult.preprocess.dataset_h5 import create_dataset


DATASET_PATHS = {
    "afewva":        "data/db_processed/AFEW-VA/AFEWVA.h5",
    "ravdess":       "data/db_processed/RAVDESS/RAVDESS.h5",
    "mosei":         "data/db_processed/MOSEI/MOSEI.h5",
    "cremad_expr":   "data/db_processed/CREMA-D/CREMAD_EXPR.h5",
    "cremad_int":    "data/db_processed/CREMA-D/CREMAD_INT.h5",
    "affwild2_expr": "data/db_processed/Aff-Wild2/AFFWILD2_EXPR.h5",
    "affwild2_va":   "data/db_processed/Aff-Wild2/AFFWILD2_VA.h5",
    "mead":          "data/db_processed/MEAD/MEAD.h5"
}


class FeatureTransformer:
    def __init__(self, config, subset):
        self.config = config
        self.subset = subset
        self.time_dims = {
            'wavlm_baseplus': config.get('time_dim_wavlm', 500),
            'clip': config.get('time_dim_frames', 300),
            'xml_roberta': config.get('time_dim_text', 120)
        }
        self.feature_dims = {
            'wavlm_baseplus': 768,
            'clip': 1024,
            'xml_roberta': 768
        }
        self.feature_list = config['feature_list']
        self.target_list = config['target_list']

    def _get_available_regions(self, mask, min_region_size=5):
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

    def _get_non_overlapping_blocks(self, regions, block_len, num_blocks, existing_blocks, min_gap=5):
        blocks = []
        for _ in range(num_blocks):
            available = []
            for r_start, r_end in regions:
                for block_start, block_end in blocks + existing_blocks:
                    r_start = max(r_start, block_end + min_gap)
                    if r_start >= r_end:
                        break
                if r_end - r_start >= block_len:
                    available.append((r_start, r_end))
            if not available:
                break
            region_idx = torch.randint(0, len(available), (1,)).item()
            region_start, region_end = available[region_idx]
            max_start = region_end - block_len
            start = region_start if max_start <= region_start else torch.randint(region_start, max_start, (1,)).item()
            end = start + block_len
            blocks.append((start, end))
        return blocks

    def _get_text_mask_indices(self, text_mask, mask_ratio=0.15):
        valid_indices = torch.nonzero(text_mask, as_tuple=True)[0].tolist()
        if not valid_indices:
            return []
        num_masks = max(1, int(len(valid_indices) * mask_ratio))
        selected = torch.randperm(len(valid_indices))[:num_masks].tolist()
        return [valid_indices[i] for i in selected]

    def __call__(self, sample):
        # First convert numpy arrays to tensors and remove .npy suffix
        new_sample = {}
        for key, value in sample.items():
            new_key = key.replace('.npy', '')

            if new_key in self.feature_list + self.target_list:
                new_sample[new_key] = value
            elif new_key in [f'{f}_mask' for f in self.feature_list + self.target_list]:
                new_sample[new_key] = value.bool()
            elif new_key in ['datasets', 'keys']:
                new_sample[new_key] = value
            else:
                continue
        sample = new_sample

        # Add missing features based on config
        for feature in self.feature_list:
            if feature not in sample:
                if feature in self.time_dims:
                    T = self.time_dims[feature]
                    D = self.feature_dims[feature]
                    sample[feature] = torch.zeros((T, D), dtype=torch.float32)
                    sample[f'{feature}_mask'] = torch.zeros(T, dtype=torch.bool)

            if self.config.get('keep_original', False):
                sample[f'original_{feature}'] = sample[feature].clone()
                sample[f'original_{feature}_mask'] = sample[f'{feature}_mask'].clone()

        # Add missing targets
        for target in self.target_list:
            if target not in sample:
                if target == 'sentiment':
                    sample[target] = torch.tensor(-5, dtype=torch.float32)  # -5 indicates no label
                    sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                if target == 'sentiment_class':
                    sample[target] = torch.tensor(-1, dtype=torch.int64)  # -1 indicates no label
                    sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                if target in ['valence', 'arousal']:
                    T = self.time_dims['clip']  # These are frame-wise
                    sample[target] = torch.full((T,), -5, dtype=torch.float32) # -5 indicates no label
                    sample[f'{target}_mask'] = torch.zeros(T, dtype=torch.bool)
                if target == 'emotion_intensity':
                    sample[target] = torch.tensor(-1, dtype=torch.int64)  # -1 indicates no label
                    sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                if target == 'emotion_class':
                    sample[target] = torch.tensor(-1, dtype=torch.int64)  # -1 indicates no label
                    sample[f'{target}_mask'] = torch.tensor(False, dtype=torch.bool)
                if target == 'emotion_class_fw':
                    T = self.time_dims['clip']  # These are frame-wise
                    sample[target] = torch.full((T,), -1, dtype=torch.int64) # -1 indicates no label
                    sample[f'{target}_mask'] = torch.zeros(T, dtype=torch.bool)
                if 'tmm_' in target:
                    feature = target.replace('tmm_', '')
                    sample[target] = sample[feature].clone()

        # Apply temporal block masking to features
        if self.config.get('block_dropout', 0) > 0 and \
            (self.subset == 'train' or (self.subset == 'valid' and any(['tmm_' in elem for elem in target]))):

            # Visual masking
            if 'clip' in self.feature_list:
                if sample['clip_mask'].any():
                    visual_regions = self._get_available_regions(sample['clip_mask'])
                    
                    visual_blocks = self._get_non_overlapping_blocks(
                        visual_regions,
                        block_len=self.config.get('block_length', 15),
                        num_blocks=self.config.get('num_block_drops', 2),
                        existing_blocks=[],
                        min_gap=self.config.get('min_gap_between_blocks', 5)
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
            if 'wavlm_baseplus' in self.feature_list:
                if sample['wavlm_baseplus_mask'].any():
                    audio_regions = self._get_available_regions(sample['wavlm_baseplus_mask'])

                    visual_blocks_audio = []
                    for vs, ve in visual_blocks:
                        as_start = int((vs / self.time_dims['clip']) * self.time_dims['wavlm_baseplus'])
                        as_end = int((ve / self.time_dims['clip']) * self.time_dims['wavlm_baseplus'])
                        visual_blocks_audio.append((as_start, as_end))

                    audio_blocks = self._get_non_overlapping_blocks(
                        audio_regions,
                        block_len=int(self.config.get('block_mask_length', 15) * (self.time_dims['wavlm_baseplus'] / self.time_dims['clip'])),
                        num_blocks=self.config.get('num_mask_blocks', 2),
                        existing_blocks=visual_blocks_audio,
                        min_gap=int(self.config.get('min_gap_between_blocks', 5) * (self.time_dims['wavlm_baseplus'] / self.time_dims['clip']))
                    )

                    audio_temporal_mask = torch.zeros_like(sample['wavlm_baseplus_mask'])
                    for s, e in audio_blocks:
                        sample['wavlm_baseplus'][s:e] = 0
                        audio_temporal_mask[s:e] = True
                    sample['tmm_wavlm_baseplus_mask'] = audio_temporal_mask
                else:
                    sample['tmm_wavlm_baseplus_mask'] = torch.zeros_like(sample['wavlm_baseplus_mask'], dtype=torch.bool)

            # Text masking (random tokens)
            if 'xml_roberta' in self.feature_list:
                if sample['xml_roberta_mask'].any():
                    text_temporal_mask = torch.zeros_like(sample['xml_roberta_mask'], dtype=torch.bool)
                    
                    # Get indices of tokens to mask (only from valid tokens)
                    mask_indices = self._get_text_mask_indices(sample['xml_roberta_mask'])
                    
                    # Apply masking
                    if mask_indices:
                        mask_indices = torch.tensor(mask_indices, dtype=torch.long)
                        sample['xml_roberta'][mask_indices] = 0
                        text_temporal_mask[mask_indices] = True
                    
                    sample['tmm_xml_roberta_mask'] = text_temporal_mask
                else:
                    sample['tmm_xml_roberta_mask'] = torch.zeros_like(sample['xml_roberta_mask'], dtype=torch.bool)

        # Apply feature corruption with Gaussian noise
        if self.config.get('feature_corruption', 0) > 0 and self.subset == 'train':
            for feature in self.feature_list:
                if feature in sample and torch.rand(1).item() < self.config['feature_corruption']:
                    # Add Gaussian noise with mean=0 and std=0.1
                    noise = torch.randn_like(sample[feature]) * 0.1
                    sample[feature] = sample[feature] + noise

        # Apply modality dropout
        if self.config.get('modality_dropout', 0) > 0 and self.subset == 'train':
            available_modalities = [f for f in self.feature_list if f in sample and f + '_mask' in sample]
            if len(available_modalities) > 1:  # Need at least 2 modalities
                # Decide which modalities to drop
                to_drop = [mod for mod in available_modalities 
                            if torch.rand(1).item() < self.config['modality_dropout']]
                
                # If all would be dropped, keep one random modality
                if len(to_drop) == len(available_modalities):
                    to_drop.remove(np.random.choice(to_drop))
                    
                # Apply dropout to selected modalities
                for mod in to_drop:
                    sample[mod].fill_(0)
                    sample[f'{mod}_mask'].fill_(False)
        
        # Apply targeted modality dropout
        if self.config.get('drop_modality', None) is not None:
            for mod in self.config['drop_modality']:
                sample[mod].fill_(0)
                sample[f'{mod}_mask'].fill_(False)

        return sample


class UnifiedDataset(Dataset):
    def __init__(self, dataset, transform_fn):
        self.dataset = dataset
        self.transform_fn = transform_fn
        
    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.transform_fn(sample)
        
    def __len__(self):
        return len(self.dataset)


class MultiDatasetModule(L.LightningDataModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.proportion_sampling = config.get('proportion_sampling', True)
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 8)
        self.dataset_config = config['datasets']
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_size = None
        self.valid_size = None
        self.test_size = None


    def _build_dataset(self, dataset_names: list[str], subset: str) -> tuple[Dataset, int]:
        """Build a unified dataset from multiple datasets
        
        Returns:
            tuple: (dataset, total_samples) where total_samples is the sum of all dataset sizes
        """
        datasets = [create_dataset(DATASET_PATHS.get(dataset_name, None), subset) for dataset_name in dataset_names]
        dataset_lengths = [len(ds) for ds in datasets]
        total_samples = sum(dataset_lengths)

        datasets = [
            UnifiedDataset(ds, transform_fn=FeatureTransformer(self.config, subset)) 
            for ds in datasets
        ]

        print(f"\n[{subset}] Total samples: {total_samples}")
        print(f"[{subset}] Epoch sizes: {list(zip(dataset_names, dataset_lengths))}\n")

        union = torch.utils.data.ConcatDataset(datasets)
       
        return union, total_samples


    def _build_balanced_dataset(self, dataset_names: list[str], subset: str) -> tuple[Dataset, int]:
        """Build a proportionally sampled dataset from multiple datasets
        
        Returns:
            tuple: (dataset, total_samples) where total_samples is the sum of all dataset sizes
        """
        # Create and transform datasets
        datasets = [create_dataset(DATASET_PATHS.get(dataset_name, None), subset) for dataset_name in dataset_names]
        dataset_lengths = [len(ds) for ds in datasets]
        total_samples = sum(dataset_lengths)
        
        # Apply transformations
        datasets = [
            UnifiedDataset(ds, transform_fn=FeatureTransformer(self.config, subset)) 
            for ds in datasets
        ]
        
        # Calculate proportions (using sqrt to reduce dominance of very large datasets)
        smooth_sizes = [math.sqrt(s) for s in dataset_lengths]
        smooth_sum = sum(smooth_sizes)
        proportions = [s / smooth_sum for s in smooth_sizes]
        
        print(f"\n[{subset}] Total samples: {total_samples}")
        print(f"[{subset}] Proportions: {list(zip(dataset_names, [round(p, 3) for p in proportions]))}")
        print(f"[{subset}] Dataset sizes: {list(zip(dataset_names, dataset_lengths))}\n")
        
        # Create a single ConcatDataset
        concat_dataset = ConcatDataset(datasets)
        
        # Create sample weights for each dataset
        sample_weights = []
        for idx, dataset in enumerate(datasets):
            # Weight for each sample in this dataset
            weight = proportions[idx] / len(dataset)
            sample_weights.extend([weight] * len(dataset))
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=total_samples,
            replacement=True
        )
        
        return concat_dataset, total_samples, sampler


    def setup(self, stage: str | None = None):
        if stage == 'fit' or stage is None:
            if self.proportion_sampling:
                self.train_dataset, self.train_size, self.train_sampler = self._build_balanced_dataset(
                    self.dataset_config['train'], 
                    subset='train'
                )
            else:
                self.train_dataset, self.train_size = self._build_dataset(
                    self.dataset_config['train'], 
                    subset='train'
                )
                self.train_sampler = None

            self.valid_dataset, self.valid_size = self._build_dataset(
                self.dataset_config['valid'], 
                subset='valid'
            )

        if stage == 'test' or stage is None:
            self.test_dataset, self.test_size = self._build_dataset(
                self.dataset_config['test'], 
                subset='test'
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            #pin_memory=True # MPS doesn't support pin_memory
            drop_last=True  # Helps with batch norm stability
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            #pin_memory=True # MPS doesn't support pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            #pin_memory=True # MPS doesn't support pin_memory
        )

    def get_class_distribution(self, subset: str):
        if subset == 'train':
            dataloader = self.train_dataloader()
        elif subset == 'valid':
            dataloader = self.val_dataloader()
        elif subset == 'test':
            dataloader = self.test_dataloader()
        else:
            raise ValueError(f"Unknown subset: {subset}")

        emotion_class_counts = torch.zeros(7, dtype=torch.int64)    # 8 unified emotion classes, but no contempt
        intensity_class_counts = torch.zeros(3, dtype=torch.int64)  # 3 unified intensity classes
        for batch in tqdm(dataloader, desc=f"Counting classes"):
            for elem in batch['emotion_class'][batch['emotion_class_mask']]:
                emotion_class_counts[int(elem)] += 1
            for elem in batch['emotion_intensity'][batch['emotion_intensity_mask']]:
                intensity_class_counts[int(elem)] += 1

        return emotion_class_counts, intensity_class_counts


if __name__ == "__main__":

    config = {
        'feature_list': ['wavlm_baseplus', 'clip'], 
        'target_list': ['emotion_class', 'emotion_intensity', 'sentiment'], 
        'datasets': {
            'train': ['ravdess', 'mosei', 'cremad_expr'], #, 'celebv_hq', 'cremad', 'meld', 'mosei', 'mead'],
            'valid': ['ravdess', 'mosei'],
            'test': ['ravdess']
        },
        'num_workers': 10,
    }

    datamodule = MultiDatasetModule(config)
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    for i, batch in enumerate(train_dataloader):
        print(f"[Batch {i}]" +
        f" ravdess: {sum([elem == 'RAVDESS' for elem in batch['datasets']]) / len(batch['datasets'])} | " +
        f" mosei: {sum([elem == 'MOSEI' for elem in batch['datasets']]) / len(batch['datasets'])} | " +
        f" cremad_expr: {sum([elem == 'CREMA-D_expr' for elem in batch['datasets']]) / len(batch['datasets'])}")
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
