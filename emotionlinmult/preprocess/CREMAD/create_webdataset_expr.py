from pathlib import Path
from tqdm import tqdm
import webdataset as wds
import numpy as np
import pandas as pd
import pickle
import json
from exordium.utils.padding import pad_or_crop_time_dim
from sklearn.model_selection import train_test_split
from emotionlinmult.preprocess import (
    CLIP_TIME_DIM, WAVLM_BASEPLUS_TIME_DIM,
    CLIP_FEATURE_DIM, WAVLM_BASEPLUS_FEATURE_DIM,
    CREMAD_EMOTION_NAME2ORIG, CREMAD_INTENSITY_NAME2ORIG
)
from emotionlinmult.preprocess.CREMAD import DB_PROCESSED


IGNORE_IDS = [
    '1076_MTI_NEU_XX',
    '1076_MTI_SAD_XX',
    '1064_TIE_SAD_XX',
    '1064_IEO_DIS_MD'
]


def save_webdataset(subset: str, features: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = sorted([
        elem.stem for elem in list(Path(DB_PROCESSED / 'videos').glob('*.mp4')) 
        if elem.stem not in IGNORE_IDS
    ]) # 7438
    participant_ids = list(set([sample_id.split('_')[0] for sample_id in sample_ids]))
    print(f'Number of samples: {len(sample_ids)}')
    print(f'Number of participants: {len(participant_ids)}')

    SUBSET_RATIO = {
        'train': 0.6,
        'valid': 0.2,
        'test': 0.2,
    }

    train_ids, valid_test_ids = train_test_split(participant_ids, test_size=SUBSET_RATIO['valid'] + SUBSET_RATIO['test'], random_state=42)
    valid_ids, test_ids = train_test_split(valid_test_ids, test_size=0.5, random_state=42)
    print(f"Train: {len(train_ids)} ({len(train_ids)/len(participant_ids):.1%})")
    print(f"Valid: {len(valid_ids)} ({len(valid_ids)/len(participant_ids):.1%})")
    print(f"Test: {len(test_ids)} ({len(test_ids)/len(participant_ids):.1%})")

    SUBSET_PARTICIPANT_IDS = {
        'train': train_ids,
        'valid': valid_ids,
        'test': test_ids
    }
    subset_ids = [sample_id for sample_id in sample_ids if sample_id.split('_')[0] in SUBSET_PARTICIPANT_IDS[subset]]

    with wds.ShardWriter(str(output_dir / f"cremad_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_id in tqdm(subset_ids, desc=f"CREMA-D {subset}"):

            clip_path = DB_PROCESSED / 'clip' / f'{sample_id}.npy'
            wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl'

            clip_full = np.load(clip_path) # (T, F)
            with open(wavlm_path, 'rb') as f:
                wavlm_full = np.array(pickle.load(f))  # (12, T_aud, F)
                wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_aud, F)

            clip, clip_mask = pad_or_crop_time_dim(clip_full, CLIP_TIME_DIM)
            wavlm, wavlm_mask = pad_or_crop_time_dim(wavlm_full, WAVLM_BASEPLUS_TIME_DIM)

            emotion_class = CREMAD_EMOTION_NAME2ORIG[sample_id.split('_')[2]]

            # Verify feature dimensions
            assert wavlm.shape == (WAVLM_BASEPLUS_TIME_DIM, WAVLM_BASEPLUS_FEATURE_DIM), \
                f"Invalid WavLM shape: {wavlm.shape}"
            assert clip.shape == (CLIP_TIME_DIM, CLIP_FEATURE_DIM), \
                f"Invalid CLIP shape: {clip.shape}"

            # Build feature dictionary
            feat_dict = {
                'clip': clip,
                'clip_mask': clip_mask,
                'wavlm_baseplus': wavlm,
                'wavlm_baseplus_mask': wavlm_mask,
            }

            # Create sample
            sample = {
                '__key__': sample_id,
                'emotion_class.npy': np.array(emotion_class, np.int64),
            }

            # Add selected features
            sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
            sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

            sink.write(sample)


if __name__ == '__main__':
    output_dir = DB_PROCESSED / "webdataset_expr"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset=subset, features=['clip', 'wavlm_baseplus'], output_dir=output_dir)