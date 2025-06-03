from pathlib import Path
from tqdm import tqdm
from enum import Enum
import webdataset as wds
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from exordium.utils.padding import pad_or_crop_time_dim


DB = Path("data/db/Aff-Wild2")
DB_PROCESSED = Path("data/db_processed/Aff-Wild2")


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

    fps = 30
    sr = 50
    clip_time_dim = analysis_window_sec * fps  # 10 sec @ 30 fps
    wavlm_baseplus_time_dim = analysis_window_sec * sr  # 10 sec @ ~50 sr

    with wds.ShardWriter(str(output_dir / f"affwild2_va_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_path in tqdm(sample_paths, total=len(sample_paths), desc=f"Aff-Wild2 VA {subset}"):
            sample_id = sample_path.stem
            annotation = get_annotation(sample_path)
            annotation_valence = np.array([v['valence'] for v in annotation.values()]) # (T,)
            annotation_arousal = np.array([v['arousal'] for v in annotation.values()]) # (T,)

            # Load video features
            clip_path = DB_PROCESSED / 'clip' / f'{sample_id}.npy'
            clip_full = np.load(clip_path) if clip_path.exists() else None  # (T_vid, F)

            # Load audio features
            wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl'
            if wavlm_path.exists():
                with open(wavlm_path, 'rb') as f:
                    wavlm_full = np.array(pickle.load(f))  # (12, T_aud, F)
                    wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_aud, F)
            else:
                wavlm_full = None

            if clip_full is None and wavlm_full is None:
                raise ValueError(f"No features found for {sample_id}.")

            num_chunks = int(np.ceil(len(annotation) / clip_time_dim))
            if num_chunks == 0:
                raise ValueError(f"No annotation for {sample_id}.")

            for chunk_idx in range(num_chunks):
                chunk_key = f"{sample_id}_{chunk_idx:03d}"

                # Process video features
                if clip_full is not None:
                    start = chunk_idx * clip_time_dim
                    end = start + clip_time_dim
                    clip_chunk = clip_full[start:end]
                    clip, clip_mask = pad_or_crop_time_dim(clip_chunk, clip_time_dim)
                else:
                    clip = np.zeros((clip_time_dim, 1024), np.float32)
                    clip_mask = np.zeros(clip_time_dim, bool)

                # Process audio features
                if wavlm_full is not None:
                    start = chunk_idx * wavlm_baseplus_time_dim
                    end = start + wavlm_baseplus_time_dim
                    wavlm_chunk = wavlm_full[start:end]
                    wavlm, wavlm_mask = pad_or_crop_time_dim(wavlm_chunk, wavlm_baseplus_time_dim)
                else:
                    wavlm = np.zeros((wavlm_baseplus_time_dim, 768), np.float32)
                    wavlm_mask = np.zeros(wavlm_baseplus_time_dim, bool)

                # Verify feature dimensions
                assert wavlm.ndim == 2 and wavlm.shape[1] == 768, \
                    f"Invalid WavLM shape: {wavlm.shape}"
                assert clip.ndim == 2 and clip.shape[1] == 1024, \
                    f"Invalid CLIP shape: {clip.shape}"

                # Build feature dictionary
                feat_dict = {
                    'clip': clip,
                    'clip_mask': clip_mask,
                    'wavlm_baseplus': wavlm,
                    'wavlm_baseplus_mask': wavlm_mask,
                    #'xml_roberta': xml_roberta, # maybe later with whisper
                    #'xml_roberta_mask': np.ones(xml_roberta.shape[0], bool), # maybe later with whisper
                }

                # Process labels
                start = chunk_idx * clip_time_dim # frame-wise, similarly to clip
                end = start + clip_time_dim

                valence_chunk = annotation_valence[start:end]
                valence, valence_mask = pad_or_crop_time_dim(valence_chunk, clip_time_dim)
                valence_mask[valence == -5] = False # filter invalid labels

                arousal_chunk = annotation_arousal[start:end]
                arousal, arousal_mask = pad_or_crop_time_dim(arousal_chunk, clip_time_dim)
                arousal_mask[arousal == -5] = False # filter invalid labels

                if valence_mask.sum() == 0 or arousal_mask.sum() == 0: continue # skip empty chunks

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
        save_webdataset(subset, ['clip', 'wavlm_baseplus'], output_dir)