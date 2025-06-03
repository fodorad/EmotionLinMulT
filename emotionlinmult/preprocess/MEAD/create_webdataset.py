from pathlib import Path
from enum import Enum
from tqdm import tqdm
import webdataset as wds
import numpy as np
import pandas as pd
import pickle
import json
from exordium.utils.padding import pad_or_crop_time_dim
from sklearn.model_selection import train_test_split


DB = Path("data/db/MEAD")
DB_PROCESSED = Path("data/db_processed/MEAD")

PARTICIPANT_IDS = sorted([elem.name for elem in list(DB.glob("*")) if len(elem.name) == 4 and elem.name[0] in ['M', 'W']])
MALE_IDS = [elem for elem in PARTICIPANT_IDS if elem[0] == 'M']
FEMALE_IDS = [elem for elem in PARTICIPANT_IDS if elem[0] == 'W']

male_ids_train, male_rest_ids = train_test_split(MALE_IDS, test_size=0.3, random_state=40)
male_ids_valid, male_ids_test = train_test_split(male_rest_ids, test_size=0.5, random_state=40)
female_ids_train, female_rest_ids = train_test_split(FEMALE_IDS, test_size=0.3, random_state=40)
female_ids_valid, female_ids_test = train_test_split(female_rest_ids, test_size=0.5, random_state=40)

SUBSET_PARTICIPANT_IDS = {
    'train': male_ids_train + female_ids_train,
    'valid': male_ids_valid + female_ids_valid,
    'test':  male_ids_test + female_ids_test,
}

EMOTION_TO_CLASS = {
    'neutral': 0,
    'angry': 1,
    'contempt': 2,
    'disgusted': 3,
    'happy': 4,
    'fear': 5,
    'sad': 6,
    'surprised': 7
}

CAMERA_POSITIONS = [
    'down',
    'front',
    'top',
    'left_30',
    'left_60',
    'right_30',
    'right_60'
]

CAMERA_TO_ID = {camera: idx for idx, camera in enumerate(CAMERA_POSITIONS)}
ID_TO_CAMERA = {idx: camera for camera, idx in CAMERA_TO_ID.items()}

SEX_TO_ID = {
    'W': 0,
    'M': 1,
}
ID_TO_SEX = {idx: sex for sex, idx in SEX_TO_ID.items()}

class OriginalMeadEmotionIntensity(Enum):
    WEAK = 1
    NORMAL = 2
    STRONG = 3

CLASS_TO_EMOTION = {v: k for k, v in EMOTION_TO_CLASS.items()}
INTENSITY_TO_CLASS = {e.name.lower(): e.value for e in OriginalMeadEmotionIntensity}
CLASS_TO_INTENSITY = {e.value: e.name.lower() for e in OriginalMeadEmotionIntensity}


def parse_mead_path(path: Path) -> dict:
    return {
        "participant_id": path.parents[4].name,                  # e.g., "M01"
        "camera_position": path.parents[2].name,                 # e.g., "front"
        "emotion_class": EMOTION_TO_CLASS[path.parents[1].name], # e.g., 4 (happy)
        "emotion_intensity": int(path.parent.name[-1]),          # e.g., 2
        "video_id": path.stem                                    # e.g., "001"
    }


def get_mead_samples(participant_ids: list[str]):
    samples = []
    for participant_id in participant_ids:

        wavlm_dir = DB_PROCESSED / participant_id / 'wavlm_baseplus'
        wavlm_paths = list(wavlm_dir.glob('*.pkl'))

        for wavlm_path in wavlm_paths:
            parts = wavlm_path.stem.split('-')
            emotion_class, emotion_intensity, clip_id = parts
            
            for camera in CAMERA_POSITIONS:
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


def save_webdataset(subset: str, features: list[str], output_dir: Path, analysis_window_sec: int = 10):

    output_dir.mkdir(parents=True, exist_ok=True)

    fps = 30
    sr = 50
    clip_time_dim = analysis_window_sec * fps  # 10 sec @ 30 fps
    wavlm_baseplus_time_dim = analysis_window_sec * sr  # 10 sec @ ~50 sr

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

            # Determine chunking
            total_frames = clip_full.shape[0]
            total_audio_frames = wavlm_full.shape[0]
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
                    'sex.npy': np.array(SEX_TO_ID[sample_dict["participant_id"][0]], np.int64),
                    'camera_id.npy': np.array(CAMERA_TO_ID[sample_dict["camera"]], np.int64),
                    'emotion_class.npy': np.array(sample_dict["emotion_class"], np.int64),
                    'emotion_intensity.npy': np.array(sample_dict["emotion_intensity"], np.int64),
                }

                sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
                sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

                sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset, ['clip', 'wavlm_baseplus'], output_dir)