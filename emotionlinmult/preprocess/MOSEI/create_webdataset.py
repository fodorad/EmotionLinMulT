from pathlib import Path
from tqdm import tqdm
import webdataset as wds
import numpy as np
import pandas as pd
import pickle
from exordium.utils.padding import pad_or_crop_time_dim


DB_PROCESSED = Path("data/db_processed/MOSEI")


def get_annotation(csv_path: str, subset: str):
    assert subset in ['train', 'valid', 'test'], f"Invalid subset: {subset}"

    df = pd.read_csv(csv_path, sep=',')

    annotation = {}
    for _, row in df.iterrows():

        if row['mode'] != subset: continue

        sample_id = f"{row['video_id']}_{row['clip_id']}"
        annotation[sample_id] = {
            'sentiment': float(row['label']),
            'text': row['text'].strip()
        }

    return annotation


def save_webdataset(subset: str, features: list[str], output_dir: Path, analysis_window_sec: int = 10):
    annotation_path = DB_PROCESSED / 'mosei_label.csv'
    annotation = get_annotation(DB_PROCESSED / 'mosei_label.csv', subset)
    subset_ids = sorted(list(annotation.keys()))
    
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_roberta_path = DB_PROCESSED / 'xml_roberta' / 'mosei_xml_roberta.pkl'
    with open(xml_roberta_path, 'rb') as f:
        xml_roberta_dict = pickle.load(f)

    fps = 30
    sr = 50
    clip_time_dim = analysis_window_sec * fps  # 10 sec @ 30 fps
    wavlm_baseplus_time_dim = analysis_window_sec * sr  # 10 sec @ ~50 sr
    xml_roberta_time_dim = 120 # from earlier experimentation

    with wds.ShardWriter(str(output_dir / f"mosei_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_id in tqdm(subset_ids, total=len(subset_ids), desc=f"MOSEI {subset}"):
            sentiment = annotation[sample_id]['sentiment']
            xml_roberta_full = xml_roberta_dict[sample_id]['xml_roberta'] # (T_3, 768)

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

            vid_frames = clip_full.shape[0]
            aud_frames = wavlm_full.shape[0]

            num_visual_chunks = (vid_frames + clip_time_dim - 1) // clip_time_dim
            num_acoustic_chunks = (aud_frames + wavlm_baseplus_time_dim - 1) // wavlm_baseplus_time_dim
            num_chunks = max(1, max(num_visual_chunks, num_acoustic_chunks))

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
                    'sentiment.npy': np.array(sentiment, np.float32),
                }

                # Add selected features
                sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
                sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

                sink.write(sample)

if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset, ['clip', 'wavlm_baseplus', 'xml_roberta'], output_dir)
