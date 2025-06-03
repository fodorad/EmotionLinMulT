from pathlib import Path
from enum import Enum
from tqdm import tqdm
import webdataset as wds
import numpy as np
import pandas as pd
import pickle
import json
from exordium.utils.padding import pad_or_crop_time_dim


DB = Path("data/db/RAVDESS")
DB_PROCESSED = Path("data/db_processed/RAVDESS")
PARTICIPANT_IDS = {
    'train': [f"{i:02d}" for i in range(1, 25)][0:16],
    'valid': [f"{i:02d}" for i in range(1, 25)][16:20],
    'test':  [f"{i:02d}" for i in range(1, 25)][20:],
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


def get_annotation(sample_id: str):
    components = sample_id.split('-')
    if len(components) != 7:
        raise ValueError(f"Sample id format is incorrect: {sample_id}")

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


def save_webdataset(subset: str, features: list[str], output_dir: Path, analysis_window_sec: int = 10):
    sample_ids_all = sorted(list(set(
        [elem.stem for elem in (DB_PROCESSED / 'wavlm_baseplus').glob('*.pkl')] + 
        [elem.stem for elem in (DB_PROCESSED / 'clip').glob('*.npy')]
    )))
    song_ids = [sample_id for sample_id in sample_ids_all if sample_id.split('-')[1] == '02']
    speech_ids = [sample_id for sample_id in sample_ids_all if sample_id.split('-')[1] == '01']
    sample_ids = [sample_id for sample_id in speech_ids if sample_id.split('-')[-1] in PARTICIPANT_IDS[subset]]

    output_dir.mkdir(parents=True, exist_ok=True)

    fps = 30
    sr = 50
    clip_time_dim = analysis_window_sec * fps  # 10 sec @ 30 fps
    wavlm_baseplus_time_dim = analysis_window_sec * sr  # 10 sec @ ~50 sr

    with wds.ShardWriter(str(output_dir / f"ravdess_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_id in tqdm(sample_ids, desc=f"RAVDESS {subset}"):
            annotation = get_annotation(sample_id)
            emotion_class = annotation["emotion_class"]
            emotion_intensity = annotation["emotion_intensity"]
            modality = annotation["modality"]

            # Load features based on modality
            clip_full = None
            wavlm_full = None

            if modality in [1, 2]:
                clip_full = np.load(DB_PROCESSED / 'clip' / f'{sample_id}.npy') # (T_1, F)

            if modality in [1, 3]:
                with open(DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl', 'rb') as f:
                    wavlm_full = np.array(pickle.load(f)) # (12, T_2, F)
                    wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_2, F)

            if clip_full is None and wavlm_full is None:
                raise ValueError(f"Sample {sample_id} has no audio and video features. Please check it manually.")

            # Determine chunking
            total_frames = clip_full.shape[0] if clip_full is not None else 0
            total_audio_frames = wavlm_full.shape[0] if wavlm_full is not None else 0
            num_visual_chunks = (total_frames + clip_time_dim - 1) // clip_time_dim
            num_acoustic_chunks = (total_audio_frames + wavlm_baseplus_time_dim - 1) // wavlm_baseplus_time_dim
            num_chunks = max(1, max(num_visual_chunks, num_acoustic_chunks))

            for chunk_idx in range(num_chunks):
                chunk_key = f"{sample_id}_{chunk_idx:03d}"

                # Process video features
                start = chunk_idx * clip_time_dim
                end = start + clip_time_dim
                if clip_full is not None:
                    clip_chunk = clip_full[start:end]
                    clip, clip_mask = pad_or_crop_time_dim(clip_chunk, clip_time_dim)
                else:
                    clip = np.zeros((clip_time_dim, 1024), np.float32)
                    clip_mask = np.zeros(clip_time_dim, bool)

                # Process audio features
                start = chunk_idx * wavlm_baseplus_time_dim
                end = start + wavlm_baseplus_time_dim
                if wavlm_full is not None:
                    wavlm_chunk = wavlm_full[start:end]
                    wavlm, wavlm_mask = pad_or_crop_time_dim(wavlm_chunk, wavlm_baseplus_time_dim)
                else:
                    wavlm = np.zeros((wavlm_baseplus_time_dim, 768), np.float32)
                    wavlm_mask = np.zeros(wavlm_baseplus_time_dim, bool)

                assert wavlm.ndim == 2 and wavlm.shape[1] == 768, \
                    f"Invalid WavLM shape: {wavlm.shape}"
                assert clip.ndim == 2 and clip.shape[1] == 1024, \
                    f"Invalid CLIP shape: {clip.shape}"

                feat_dict = {
                    'clip': clip,
                    'clip_mask': clip_mask,
                    'wavlm_baseplus': wavlm,
                    'wavlm_baseplus_mask': wavlm_mask,
                }

                sample = {
                    '__key__': chunk_key,
                    'emotion_class.npy': np.array(emotion_class, np.int64),
                    'emotion_intensity.npy': np.array(emotion_intensity, np.int64),
                }

                sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
                sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

                sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset, ['clip', 'wavlm_baseplus'], output_dir)