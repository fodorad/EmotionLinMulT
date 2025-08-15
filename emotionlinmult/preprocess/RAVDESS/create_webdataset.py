import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import webdataset as wds
from exordium.utils.padding import pad_or_crop_time_dim
from emotionlinmult.preprocess import (
    CLIP_TIME_DIM, WAVLM_BASEPLUS_TIME_DIM, CLIP_FEATURE_DIM, WAVLM_BASEPLUS_FEATURE_DIM
)
from emotionlinmult.preprocess.RAVDESS import DB, DB_PROCESSED, PARTICIPANT_IDS


def get_annotation(sample_id: str):
    components = sample_id.split('-')

    if len(components) != 7:
        raise ValueError(f"Sample id format is incorrect: {sample_id}")

    annotation = {
        "modality": int(components[0]), # 1: AV; 2: V; 3: A
        "vocal_channel": int(components[1]), # 1: speech; 2: song
        "emotion_class": int(components[2]), # Original 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
        "emotion_intensity": int(components[3]), # Original 1: normal; 2: strong
        "statement": int(components[4]), # 1: "Kids are talking by the door", 2: "Dogs are sitting by the door"
        "repetition": int(components[5]), # 1: 1st repetition, 2: 2nd repetition
        "actor": int(components[6]), # 1 to 24. Odd numbered actors are male, even numbered actors are female
    }

    return annotation


def save_webdataset(subset: str, features: list[str], output_dir: Path):
    video_paths = sorted(
        [elem for elem in list(DB.glob('**/*.mp4')) \
            if int(elem.stem.split("-")[0]) == 1 and \
               int(elem.stem.split("-")[1]) == 1] # 0: AV; 1: speech;
    )
    sample_ids = [elem.stem for elem in video_paths if elem.stem.split('-')[-1] in PARTICIPANT_IDS[subset]]

    output_dir.mkdir(parents=True, exist_ok=True)
    with wds.ShardWriter(str(output_dir / f"ravdess_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_id in tqdm(sample_ids, desc=f"RAVDESS {subset}"):
            annotation = get_annotation(sample_id)
            emotion_class = annotation["emotion_class"]
            emotion_intensity = annotation["emotion_intensity"]

            clip_full = np.load(DB_PROCESSED / 'clip' / f'{sample_id}.npy') # (T_clip, CLIP_FEATURE_DIM)

            with open(DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl', 'rb') as f:
                wavlm_full = np.array(pickle.load(f)) # (12, T_wavlm, WAVLM_BASEPLUS_FEATURE_DIM)
                wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_wavlm, WAVLM_BASEPLUS_FEATURE_DIM)

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
                'emotion_class.npy': np.array(emotion_class, np.int64),
                'emotion_intensity.npy': np.array(emotion_intensity, np.int64),
            }

            sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
            sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

            sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset=subset, features=['clip', 'wavlm_baseplus'], output_dir=output_dir)