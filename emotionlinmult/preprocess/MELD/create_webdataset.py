from pathlib import Path
from tqdm import tqdm
import webdataset as wds
import numpy as np
import pandas as pd
import pickle
from exordium.utils.padding import pad_or_crop_time_dim


DB = Path("data/db/MELD")
DB_PROCESSED = Path("data/db_processed/MELD")

IGNORE_DICT = {
    "train": ["dia125_utt3"], # broken video
    "valid": ["dia110_utt7"], # missing video
    "test": []
}

def save_webdataset(subset: str, features: list[str], output_dir: Path, analysis_window_sec: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_roberta_path = DB_PROCESSED / 'xml_roberta' / 'meld_xml_roberta.pkl'
    with open(xml_roberta_path, 'rb') as f:
        xml_roberta_dict = pickle.load(f)

    subset_data = xml_roberta_dict[subset]
    subset_ids = sorted(list(subset_data.keys()))

    fps = 30
    sr = 50
    clip_time_dim = analysis_window_sec * fps  # 10 sec @ 30 fps
    wavlm_baseplus_time_dim = analysis_window_sec * sr  # 10 sec @ ~50 sr
    xml_roberta_time_dim = 120 # from earlier experimentation

    with wds.ShardWriter(str(output_dir / f"meld_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_id in tqdm(subset_ids, total=len(subset_ids), desc=f"MELD {subset}"):

            if sample_id in IGNORE_DICT[subset]: continue

            sentiment = subset_data[sample_id]['sentiment']
            emotion_class = subset_data[sample_id]['emotion_class']
            xml_roberta_full = subset_data[sample_id]['xml_roberta'] # (120, 768)

            # Load video features
            clip_path = DB_PROCESSED / 'clip' / subset / f'{sample_id}.npy'
            clip_full = np.load(clip_path)  # (T_vid, F)

            # Load audio features
            wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / subset / f'{sample_id}.pkl'
            with open(wavlm_path, 'rb') as f:
                wavlm_full = np.array(pickle.load(f))  # (12, T_aud, F)
                wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_aud, F)

            vid_frames = clip_full.shape[0]
            aud_frames = wavlm_full.shape[0]

            num_visual_chunks = (vid_frames + clip_time_dim - 1) // clip_time_dim
            num_acoustic_chunks = (aud_frames + wavlm_baseplus_time_dim - 1) // wavlm_baseplus_time_dim
            num_chunks = max(1, max(num_visual_chunks, num_acoustic_chunks))

            for chunk_idx in range(num_chunks):
                chunk_key = f"{sample_id}_{chunk_idx:03d}"

                # Process video features
                start = chunk_idx * clip_time_dim
                end = start + clip_time_dim
                clip_chunk = clip_full[start:end]
                clip, clip_mask = pad_or_crop_time_dim(clip_chunk, clip_time_dim)

                # Process audio features
                start = chunk_idx * wavlm_baseplus_time_dim
                end = start + wavlm_baseplus_time_dim
                wavlm_chunk = wavlm_full[start:end]
                wavlm, wavlm_mask = pad_or_crop_time_dim(wavlm_chunk, wavlm_baseplus_time_dim)

                # Process text features
                xml_roberta, xml_roberta_mask = pad_or_crop_time_dim(xml_roberta_full, xml_roberta_time_dim)

                # Verify feature dimensions
                assert wavlm.ndim == 2 and wavlm.shape[1] == 768, \
                    f"Invalid WavLM shape: {wavlm.shape}"
                assert clip.ndim == 2 and clip.shape[1] == 1024, \
                    f"Invalid CLIP shape: {clip.shape}"
                assert xml_roberta.ndim == 2 and xml_roberta.shape[1] == 768, \
                    f"Invalid XML RoBERTa shape: {xml_roberta.shape}"

                # Build feature dictionary
                feat_dict = {
                    'clip': clip,
                    'clip_mask': clip_mask,
                    'wavlm_baseplus': wavlm,
                    'wavlm_baseplus_mask': wavlm_mask,
                    'xml_roberta': xml_roberta,
                    'xml_roberta_mask': xml_roberta_mask,
                }

                # Create sample
                sample = {
                    '__key__': chunk_key,
                    'sentiment_class.npy': np.array(sentiment, np.int64),
                    'emotion_class.npy': np.array(emotion_class, np.int64),
                }

                # Add selected features
                sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
                sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

                sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset, ['clip', 'wavlm_baseplus', 'xml_roberta'], output_dir)