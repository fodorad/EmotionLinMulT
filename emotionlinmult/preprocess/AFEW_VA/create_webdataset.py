from pathlib import Path
import webdataset as wds
import pickle
import json
import numpy as np
from tqdm import tqdm
from exordium.utils.padding import pad_or_crop_time_dim


DB = Path("data/db/AFEW-VA")
DB_PROCESSED = Path("data/db_processed/AFEW-VA")


SUBSET_IDS = {
    'train': np.arange(1, 421),   # 70%
    'valid': np.arange(421, 511), # 15%
    'test': np.arange(511, 601),  # 15%
}


def load_sample_gt(json_path: str):
    with open(json_path, "r") as f:
        sample_annotations = json.load(f)

    arousal = np.array([ann['arousal'] for ann in sample_annotations["frames"].values()], dtype=np.float32)
    valence = np.array([ann['valence'] for ann in sample_annotations["frames"].values()], dtype=np.float32)

    return {
        'valence': valence, # shape: (n_frames,)
        'arousal': arousal, # shape: (n_frames,)
    }


def save_webdataset(subset: str, features: list[str], output_dir: Path, analysis_window_sec: int = 10):
    subset_ids = SUBSET_IDS[subset]
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_time_dim = analysis_window_sec * 30  # fps

    with wds.ShardWriter(str(output_dir / f"afewva_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_id in tqdm(subset_ids, desc=f"AFEW-VA {subset}"):
            sample_id_str = f"{sample_id:03d}"

            with open(DB_PROCESSED / 'clip' / f'{sample_id_str}.pkl', 'rb') as f:
                _, clip_full = pickle.load(f)

            with open(DB_PROCESSED / 'clip_face' / f'{sample_id_str}.pkl', 'rb') as f:
                _, clip_face_full = pickle.load(f)

            annotations = load_sample_gt(DB / 'samples' / sample_id_str / f'{sample_id_str}.json')

            assert clip_full.shape[0] == clip_face_full.shape[0], f"clip and clip_face have different number of frames for sample {sample_id_str}"
            assert clip_full.shape[0] == annotations['valence'].shape[0], f"clip and valence have different number of frames for sample {sample_id_str}"
            assert clip_full.shape[0] == annotations['arousal'].shape[0], f"clip and arousal have different number of frames for sample {sample_id_str}"

            num_chunks = int(np.ceil(annotations['valence'].shape[0] / clip_time_dim))
            if num_chunks == 0:
                raise ValueError(f"No annotation for {sample_id_str}.")

            for chunk_idx in range(num_chunks):
                chunk_key = f"{sample_id_str}_{chunk_idx:03d}"

                # Process video features
                start = chunk_idx * clip_time_dim
                end = (chunk_idx + 1) * clip_time_dim

                clip_chunk = clip_full[start:end]
                clip_face_chunk = clip_face_full[start:end]

                clip, clip_mask = pad_or_crop_time_dim(clip_chunk, clip_time_dim)
                clip_face, clip_face_mask = pad_or_crop_time_dim(clip_face_chunk, clip_time_dim)

                # Verify feature dimensions
                assert clip.ndim == 2 and clip.shape[1] == 1024, \
                    f"Invalid CLIP shape: {clip.shape}"
                assert clip_face.ndim == 2 and clip_face.shape[1] == 1024, \
                    f"Invalid CLIP (face) shape: {clip_face.shape}"

                feat_dict = {
                    'clip': clip, # (T, F)
                    'clip_mask': clip_mask, # (T,)
                    'clip_face': clip_face, # (T, F)
                    'clip_face_mask': clip_face_mask, # (T,)
                }

                # annotations
                valence_chunk = annotations['valence'][start:end]
                arousal_chunk = annotations['arousal'][start:end]

                valence, valence_mask = pad_or_crop_time_dim(valence_chunk, clip_time_dim)
                arousal, arousal_mask = pad_or_crop_time_dim(arousal_chunk, clip_time_dim)

                if valence_mask.sum() == 0 or arousal_mask.sum() == 0: continue # skip empty chunks

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
        save_webdataset(subset, ['clip_face'], output_dir)