from pathlib import Path
import webdataset as wds
import pickle
import json
import numpy as np
from tqdm import tqdm
from exordium.utils.padding import pad_or_crop_time_dim
from emotionlinmult.preprocess import CLIP_TIME_DIM, CLIP_FEATURE_DIM
from emotionlinmult.preprocess.AFEWVA import DB, DB_PROCESSED, SUBSET_IDS


def load_sample_gt(json_path: str):
    with open(json_path, "r") as f:
        sample_annotations = json.load(f)

    arousal = np.array([ann['arousal'] for ann in sample_annotations["frames"].values()], dtype=np.float32)
    valence = np.array([ann['valence'] for ann in sample_annotations["frames"].values()], dtype=np.float32)

    return {
        'valence': valence, # shape: (n_frames,)
        'arousal': arousal, # shape: (n_frames,)
    }


def save_webdataset(subset: str, features: list[str], output_dir: Path):
    subset_ids = SUBSET_IDS[subset]
    output_dir.mkdir(parents=True, exist_ok=True)

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

            # annotations
            valence, valence_mask = pad_or_crop_time_dim(annotations['valence'], CLIP_TIME_DIM)
            arousal, arousal_mask = pad_or_crop_time_dim(annotations['arousal'], CLIP_TIME_DIM)
            if valence_mask.sum() == 0 or arousal_mask.sum() == 0: 
                raise ValueError(f"No annotation for {sample_id_str}.")

            # Process video features
            clip, clip_mask = pad_or_crop_time_dim(clip_full, CLIP_TIME_DIM)
            clip_face, clip_face_mask = pad_or_crop_time_dim(clip_face_full, CLIP_TIME_DIM)

            # Verify feature dimensions
            assert clip.shape == (CLIP_TIME_DIM, CLIP_FEATURE_DIM), \
                f"Invalid CLIP shape: {clip.shape}"
            assert clip_face.shape == (CLIP_TIME_DIM, CLIP_FEATURE_DIM), \
                f"Invalid CLIP (face) shape: {clip_face.shape}"

            feat_dict = {
                'clip': clip, # (T, F)
                'clip_mask': clip_mask, # (T,)
                'clip_face': clip_face, # (T, F)
                'clip_face_mask': clip_face_mask, # (T,)
            }

            sample = {
                '__key__': sample_id_str,
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
        save_webdataset(subset=subset, features=['clip'], output_dir=output_dir)