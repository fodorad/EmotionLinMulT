from pathlib import Path
from tqdm import tqdm
import webdataset as wds
import numpy as np
import pandas as pd
import pickle
from exordium.utils.padding import pad_or_crop_time_dim
from emotionlinmult.preprocess import (
    CLIP_TIME_DIM, WAVLM_BASEPLUS_TIME_DIM, XML_ROBERTA_TIME_DIM,
    CLIP_FEATURE_DIM, WAVLM_BASEPLUS_FEATURE_DIM, XML_ROBERTA_FEATURE_DIM
)
from emotionlinmult.preprocess.MOSEI import DB_PROCESSED


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


def save_webdataset(subset: str, features: list[str], output_dir: Path):
    annotation_path = DB_PROCESSED / 'mosei_label.csv'
    annotation = get_annotation(DB_PROCESSED / 'mosei_label.csv', subset)
    subset_ids = sorted(list(annotation.keys()))
    
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_roberta_path = DB_PROCESSED / 'xml_roberta' / 'mosei_xml_roberta.pkl'
    with open(xml_roberta_path, 'rb') as f:
        xml_roberta_dict = pickle.load(f)

    with wds.ShardWriter(str(output_dir / f"mosei_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_id in tqdm(subset_ids, total=len(subset_ids), desc=f"MOSEI {subset}"):
            sentiment = annotation[sample_id]['sentiment']
            xml_roberta_full = xml_roberta_dict[sample_id]['xml_roberta']  # (T_text, 768)

            clip_path = DB_PROCESSED / 'clip' / f'{sample_id}.npy'
            wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl'

            if not clip_path.exists() or not wavlm_path.exists():
                continue               

            clip_full = np.load(clip_path)  # (T_vid, 1024)

            with open(wavlm_path, 'rb') as f:
                wavlm_full = np.array(pickle.load(f))  # (12, T_aud, F)
                wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_aud, 768)

            assert clip_full.ndim == 2 and clip_full.shape[1] == CLIP_FEATURE_DIM, \
                f"Invalid CLIP shape: {clip_full.shape}"
            assert wavlm_full.ndim == 2 and wavlm_full.shape[1] == WAVLM_BASEPLUS_FEATURE_DIM, \
                f"Invalid WavLM shape: {wavlm_full.shape}"
            assert xml_roberta_full.ndim == 2 and xml_roberta_full.shape[1] == XML_ROBERTA_FEATURE_DIM, \
                f"Invalid XML RoBERTa shape: {xml_roberta_full.shape}"

            clip, clip_mask = pad_or_crop_time_dim(clip_full, CLIP_TIME_DIM)
            wavlm, wavlm_mask = pad_or_crop_time_dim(wavlm_full, WAVLM_BASEPLUS_TIME_DIM)
            xml_roberta, xml_roberta_mask = pad_or_crop_time_dim(xml_roberta_full, XML_ROBERTA_TIME_DIM)

            assert wavlm.ndim == 2 and wavlm.shape == (WAVLM_BASEPLUS_TIME_DIM, WAVLM_BASEPLUS_FEATURE_DIM), \
                f"Invalid WavLM shape: {wavlm.shape}"
            assert clip.ndim == 2 and clip.shape == (CLIP_TIME_DIM, CLIP_FEATURE_DIM), \
                f"Invalid CLIP shape: {clip.shape}"
            assert xml_roberta.ndim == 2 and xml_roberta.shape == (XML_ROBERTA_TIME_DIM, XML_ROBERTA_FEATURE_DIM), \
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
                '__key__': sample_id,
                'sentiment.npy': np.array(sentiment, np.float32),
            }

            # Add selected features
            sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
            sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

            sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['train', 'valid', 'test']:
        save_webdataset(subset=subset, features=['clip', 'wavlm_baseplus', 'xml_roberta'], output_dir=output_dir)
