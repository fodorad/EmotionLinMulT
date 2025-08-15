from pathlib import Path
from enum import Enum
from tqdm import tqdm
import webdataset as wds
import numpy as np
import pandas as pd
import pickle
import json
from exordium.utils.padding import pad_or_crop_time_dim
from emotionlinmult.preprocess import (
    CLIP_TIME_DIM, WAVLM_BASEPLUS_TIME_DIM, 
    CLIP_FEATURE_DIM, WAVLM_BASEPLUS_FEATURE_DIM,
    MEAD_CAMERA_ORIG2ID
)
from emotionlinmult.preprocess.MEAD import (
    DB, DB_PROCESSED, SUBSET_PARTICIPANT_IDS,
    parse_mead_visual_path
)


def get_mead_samples(participant_ids: list[str]):
    samples = []
    for participant_id in participant_ids:

        wavlm_dir = DB_PROCESSED / participant_id / 'wavlm_baseplus'
        wavlm_paths = list(wavlm_dir.glob('*.pkl'))

        for wavlm_path in wavlm_paths:
            parts = wavlm_path.stem.split('-')
            emotion_class, emotion_intensity, clip_id = parts
            
            for camera in MEAD_CAMERA_ORIG2ID.keys():
                clip_path = DB_PROCESSED / participant_id / 'clip' / f'{camera}-{emotion_class}-{emotion_intensity}-{clip_id}.npy'
                
                if clip_path.exists():
                    samples.append({
                        'participant_id': participant_id,
                        'emotion_class': emotion_class,
                        'emotion_intensity': emotion_intensity,
                        'clip_id': clip_id,
                        'camera': camera,
                        'wavlm_path': wavlm_path,
                        'clip_path': clip_path
                    })
    return samples


def save_webdataset(subset: str, features: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_ids = SUBSET_PARTICIPANT_IDS[subset]
    sample_dicts = get_mead_samples(participant_ids)

    with wds.ShardWriter(str(output_dir / f"mead_{subset}_%06d.tar"), maxcount=1000) as sink:

        for sample_dict in tqdm(sample_dicts, desc=f"MEAD {subset}"):
            sample_id = f'{sample_dict["participant_id"]}_{sample_dict["camera"]}_{sample_dict["emotion_class"]}_{sample_dict["emotion_intensity"]}_{sample_dict["clip_id"]}'

            # Load clip
            clip_full = np.load(sample_dict['clip_path']) # (T, 1024)

            # Load wavlm
            with open(sample_dict['wavlm_path'], 'rb') as f:
                wavlm_full = np.array(pickle.load(f)) # (12, T, 768)
                wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0) # (T, 768)

            clip, clip_mask = pad_or_crop_time_dim(clip_full, CLIP_TIME_DIM)
            wavlm, wavlm_mask = pad_or_crop_time_dim(wavlm_full, WAVLM_BASEPLUS_TIME_DIM)

            assert wavlm.ndim == 2 and wavlm.shape == (WAVLM_BASEPLUS_TIME_DIM, WAVLM_BASEPLUS_FEATURE_DIM), \
                f"Invalid WavLM shape: {wavlm.shape}"
            assert clip.ndim == 2 and clip.shape == (CLIP_TIME_DIM, CLIP_FEATURE_DIM), \
                f"Invalid CLIP shape: {clip.shape}"

            feat_dict = {
                'clip': clip,
                'clip_mask': clip_mask,
                'wavlm_baseplus': wavlm,
                'wavlm_baseplus_mask': wavlm_mask,
            }

            sample = {
                '__key__': sample_id,
                'camera_id.npy': np.array(MEAD_CAMERA_ORIG2ID[sample_dict["camera"]], np.int64),
                'emotion_class.npy': np.array(sample_dict["emotion_class"], np.int64),
                'emotion_intensity.npy': np.array(sample_dict["emotion_intensity"], np.int64),
            }

            sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
            sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

            sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset=subset, features=['clip', 'wavlm_baseplus'], output_dir=output_dir)