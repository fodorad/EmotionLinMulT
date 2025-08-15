from pathlib import Path
from tqdm import tqdm
from enum import Enum
import webdataset as wds
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from exordium.utils.padding import pad_or_crop_time_dim
from emotionlinmult.preprocess import (
    CLIP_TIME_DIM,
    WAVLM_BASEPLUS_TIME_DIM,
    CLIP_FEATURE_DIM,
    WAVLM_BASEPLUS_FEATURE_DIM,
)
from emotionlinmult.preprocess.AffWild2 import DB, DB_PROCESSED


def get_annotation(file_path: str | Path):

    def _parse_va(lines):
        header = lines[0].split(',')
        annotations = {}
        for idx, line in enumerate(lines[1:]):
            frame_id = f"{idx:06d}"
            valence, arousal = map(float, line.split(','))
            annotations[frame_id] = {
                'valence': valence,
                'arousal': arousal
            }
        return annotations

    with open(str(file_path), 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    return _parse_va(lines)


def save_webdataset(subset: str, features: list[str], output_dir: Path, analysis_window_sec: int = 10):
    """Aff-Wild2 VA WebDataset"""

    annotation_dir = DB / '6th_ABAW_Annotations' / 'VA_Estimation_Challenge'

    paths_train = sorted(list((Path(annotation_dir) / 'Train_Set').glob('*.txt')))
    paths_train, paths_valid = train_test_split(paths_train, test_size=0.15, random_state=42)
    paths_test = sorted(list((Path(annotation_dir) / 'Validation_Set').glob('*.txt')))

    sample_paths = {
        'train': paths_train,
        'valid': paths_valid,
        'test': paths_test
    }[subset]

    output_dir.mkdir(parents=True, exist_ok=True)

    with wds.ShardWriter(str(output_dir / f"affwild2_va_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_path in tqdm(sample_paths, total=len(sample_paths), desc=f"Aff-Wild2 VA {subset}"):
            sample_id = sample_path.stem
            annotation = get_annotation(sample_path)
            annotation_valence = np.array([v['valence'] for v in annotation.values()]) # (T,)
            annotation_arousal = np.array([v['arousal'] for v in annotation.values()]) # (T,)

            # Load video features
            clip_path = DB_PROCESSED / 'clip' / f'{sample_id}.npy'
            clip_full = np.load(clip_path)  # (T_vid, F)

            # Load audio features
            wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl'
            if wavlm_path.exists():
                with open(wavlm_path, 'rb') as f:
                    wavlm_full = np.array(pickle.load(f))  # (12, T_aud, F)
                    wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_aud, F)
            else:
                wavlm_full = None

            num_chunks = int(np.ceil(len(annotation) / CLIP_TIME_DIM))
            if num_chunks == 0:
                raise ValueError(f"No annotation for {sample_id}.")

            for chunk_idx in range(num_chunks):
                chunk_key = f"{sample_id}_{chunk_idx:05d}"

                # Process labels
                start = chunk_idx * CLIP_TIME_DIM # frame-wise, similarly to clip
                end = start + CLIP_TIME_DIM

                valence_chunk = annotation_valence[start:end]
                valence, valence_mask = pad_or_crop_time_dim(valence_chunk, CLIP_TIME_DIM)
                valence_mask[valence == -5] = False # filter invalid labels
                arousal_chunk = annotation_arousal[start:end]
                arousal, arousal_mask = pad_or_crop_time_dim(arousal_chunk, CLIP_TIME_DIM)
                arousal_mask[arousal == -5] = False # filter invalid labels
                if valence_mask.sum() == 0 or arousal_mask.sum() == 0: continue # skip empty chunks

                # Process video features
                start = chunk_idx * CLIP_TIME_DIM
                end = start + CLIP_TIME_DIM
                clip_chunk = clip_full[start:end]
                clip, clip_mask = pad_or_crop_time_dim(clip_chunk, CLIP_TIME_DIM)

                # Process audio features
                if wavlm_full is not None:
                    start = chunk_idx * WAVLM_BASEPLUS_TIME_DIM
                    end = start + WAVLM_BASEPLUS_TIME_DIM
                    wavlm_chunk = wavlm_full[start:end]
                    wavlm, wavlm_mask = pad_or_crop_time_dim(wavlm_chunk, WAVLM_BASEPLUS_TIME_DIM)
                else:
                    wavlm = np.zeros((WAVLM_BASEPLUS_TIME_DIM, WAVLM_BASEPLUS_FEATURE_DIM), np.float32)
                    wavlm_mask = np.zeros(WAVLM_BASEPLUS_TIME_DIM, bool)

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
                    'wavlm_baseplus_mask': wavlm_mask
                }

                # Create sample
                sample = {
                    '__key__': chunk_key,
                    'valence.npy': np.array(valence, np.float32), # (T_vid,)
                    'valence_mask.npy': valence_mask, # (T_vid,)
                    'arousal.npy': np.array(arousal, np.float32), # (T_vid,)
                    'arousal_mask.npy': arousal_mask, # (T_vid,)
                }

                # Add selected features
                sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
                sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

                sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset_va"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset=subset, features=['clip', 'wavlm_baseplus'], output_dir=output_dir)