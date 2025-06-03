from pathlib import Path
import webdataset as wds
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from exordium.utils.padding import pad_or_crop_time_dim


DB = Path("data/db/VEATIC")
DB_PROCESSED = Path("data/db_processed/VEATIC")


def load_csv_to_tensor(file_path):
    df = pd.read_csv(str(file_path), header=None)
    arr = df.values.astype(np.float32)
    return arr


def load_sample_gt(video_id: int):
    valence_file = DB / 'rating_averaged' / f"{video_id}_valence.csv"
    arousal_file = DB / 'rating_averaged' / f"{video_id}_arousal.csv"
    valence = load_csv_to_tensor(valence_file)
    arousal = load_csv_to_tensor(arousal_file)
    return {
        'valence': valence[:, 0], # (T,) and range is [-1, 1]
        'arousal': arousal[:, 0], # (T,) and range is [-1, 1]
    }


def save_webdataset(subset: str, features: list[str], output_dir: Path, analysis_window_sec: int = 10):
    video_paths = sorted(list((DB / 'videos').glob('*.mp4')))
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_time_dim = analysis_window_sec * 30 # fps

    with wds.ShardWriter(str(output_dir / f"veatic_{subset}_%06d.tar"), maxcount=1000) as sink:
        for video_path in tqdm(video_paths, desc=f"VEATIC {subset}"):
            sample_id = f"{video_path.stem}"

            clip_full = np.load(str(DB_PROCESSED / 'clip' / f'{sample_id}.npy')) # (T, F)
            annotations = load_sample_gt(sample_id) # {'valence': (T,), 'arousal': (T,)}

            assert clip_full.shape[0] == annotations['valence'].shape[0], f"clip and valence have different number of frames for sample {sample_id_str}. {clip_full.shape[0]} vs {annotations['valence'].shape[0]}"
            assert clip_full.shape[0] == annotations['arousal'].shape[0], f"clip and arousal have different number of frames for sample {sample_id_str}. {clip_full.shape[0]} vs {annotations['arousal'].shape[0]}"

            num_frames = clip_full.shape[0]

            # Calculate split indices for this video
            trainval_end = int(num_frames * 0.7)
            valid_start = int(trainval_end * 0.85)
            valid_end = trainval_end
            test_start = trainval_end

            if subset == 'train':
                chunk_range = (0, valid_start)
            elif subset == 'valid':
                chunk_range = (valid_start, valid_end)
            elif subset == 'test':
                chunk_range = (test_start, num_frames)
            else:
                raise ValueError(f"Unknown subset: {subset}")

            # Compute chunk indices for this subset
            start_frame, end_frame = chunk_range
            first_chunk = start_frame // clip_time_dim
            last_chunk = (end_frame - 1) // clip_time_dim

            for chunk_idx in range(first_chunk, last_chunk + 1):

                chunk_key = f"{sample_id}_{chunk_idx:03d}"

                chunk_start = chunk_idx * clip_time_dim
                chunk_end = (chunk_idx + 1) * clip_time_dim

                # Only process chunks that overlap with the current subset range
                if chunk_end <= start_frame or chunk_start >= end_frame:
                    continue

                # Calculate overlap within the chunk
                local_start = max(chunk_start, start_frame)
                local_end = min(chunk_end, end_frame)

                # Slice the frames for this chunk
                clip_chunk = clip_full[local_start:local_end]
                clip, clip_mask = pad_or_crop_time_dim(clip_chunk, clip_time_dim)

                valence_chunk = annotations['valence'][local_start:local_end]
                arousal_chunk = annotations['arousal'][local_start:local_end]
                valence, valence_mask = pad_or_crop_time_dim(valence_chunk, clip_time_dim)
                arousal, arousal_mask = pad_or_crop_time_dim(arousal_chunk, clip_time_dim)

                if valence_mask.sum() == 0 or arousal_mask.sum() == 0:
                    raise ValueError(f"Empty chunk for {chunk_key}.")

                assert clip.ndim == 2 and clip.shape[1] == 1024, f"Invalid CLIP shape: {clip.shape}"

                feat_dict = {
                    'clip': clip, # (T, F)
                    'clip_mask': clip_mask, # (T,)
                }

                sample = {
                    '__key__': chunk_key,
                    'valence.npy': valence, # (T,)
                    'valence_mask.npy': valence_mask, # (T,)
                    'arousal.npy': arousal, # (T,)
                    'arousal_mask.npy': arousal_mask, # (T,)
                }

                sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
                sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

                sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset, ['clip'], output_dir)