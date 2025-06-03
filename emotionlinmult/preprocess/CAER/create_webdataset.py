from pathlib import Path
from enum import Enum
from tqdm import tqdm
import webdataset as wds
import numpy as np
import pandas as pd
import pickle
import json
from exordium.utils.padding import pad_or_crop_time_dim


DB = Path("data/db/CAER")
DB_PROCESSED = Path("data/db_processed/CAER")

class OriginalCaerEmotionClass(Enum):
    Anger = 0
    Disgust = 1
    Fear = 2
    Happy = 3
    Neutral = 4
    Sad = 5
    Surprise = 6

EMOTION_TO_CLASS = {e.name.lower(): e.value for e in OriginalCaerEmotionClass}
CLASS_TO_EMOTION = {e.value: e.name.lower() for e in OriginalCaerEmotionClass}


def save_webdataset(subset: str, features: list[str], output_dir: Path, analysis_window_sec: int = 10):
    video_paths = sorted(list([elem for elem in DB.glob('**/*.avi') if elem.parent.parent.name == subset]))

    output_dir.mkdir(parents=True, exist_ok=True)

    xml_roberta_path = DB_PROCESSED / 'xml_roberta' / 'caer_xml_roberta.pkl'
    with open(xml_roberta_path, 'rb') as f:
        xml_roberta_dict = pickle.load(f)

    fps = 30
    sr = 50
    clip_time_dim = analysis_window_sec * fps  # 10 sec @ 30 fps
    wavlm_baseplus_time_dim = analysis_window_sec * sr  # 10 sec @ ~50 sr
    xml_roberta_time_dim = 120 # from earlier experimentation

    with wds.ShardWriter(str(output_dir / f"caer_{subset[:5]}_%06d.tar"), maxcount=1000) as sink:
        for video_path in tqdm(video_paths, desc=f"CAER {subset}"):
            label = Path(video_path).parent.name
            clip_id = Path(video_path).stem
            sample_id = f"{subset}_{label}_{clip_id}"

            clip_path = DB_PROCESSED / 'clip' / subset /  label / f'{clip_id}.npy'
            clip_full = np.load(clip_path) # (T_1, F)

            wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / subset /  label / f'{clip_id}.pkl'
            with open(wavlm_path, 'rb') as f:
                wavlm_full = np.array(pickle.load(f)) # (12, T_2, F)
                wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_2, F)

            xml_roberta_full = xml_roberta_dict[sample_id]['xml_roberta'] # (T_3, 768)

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
                    wavlm = np.zeros((12, wavlm_baseplus_time_dim, 768), np.float32)
                    wavlm_mask = np.zeros(wavlm_baseplus_time_dim, bool)

                # Process text features
                xml_roberta, xml_roberta_mask = pad_or_crop_time_dim(xml_roberta_full, xml_roberta_time_dim)

                # Verify feature dimensions
                assert wavlm.ndim == 2 and wavlm.shape[1] == 768, \
                    f"Invalid WavLM shape: {wavlm.shape}"
                assert clip.ndim == 2 and clip.shape[1] == 1024, \
                    f"Invalid CLIP shape: {clip.shape}"
                assert xml_roberta.ndim == 2 and xml_roberta.shape[1] == 768, \
                    f"Invalid XML RoBERTa shape: {xml_roberta.shape}"

                feat_dict = {
                    'clip': clip,
                    'clip_mask': clip_mask,
                    'wavlm_baseplus': wavlm,
                    'wavlm_baseplus_mask': wavlm_mask,
                    'xml_roberta': xml_roberta,
                    'xml_roberta_mask': xml_roberta_mask,
                }

                sample = {
                    '__key__': chunk_key,
                    'emotion_class.npy': np.array(EMOTION_TO_CLASS[label.lower()], np.int64)
                }

                sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
                sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

                sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['train', 'validation', 'test']:
        save_webdataset(subset, ['clip', 'wavlm_baseplus', 'xml_roberta'], output_dir)