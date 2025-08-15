from pathlib import Path
from tqdm import tqdm
import webdataset as wds
import numpy as np
import pandas as pd
import pickle
from exordium.utils.padding import pad_or_crop_time_dim
from emotionlinmult.preprocess import (
    CLIP_FEATURE_DIM, WAVLM_BASEPLUS_FEATURE_DIM, XML_ROBERTA_FEATURE_DIM,
    CLIP_TIME_DIM, WAVLM_BASEPLUS_TIME_DIM, XML_ROBERTA_TIME_DIM
)
from emotionlinmult.preprocess.MELD import DB, DB_PROCESSED


IGNORE_DICT = {
    "train": ["dia125_utt3"], # broken video
    "valid": ["dia110_utt7"], # missing video
    "test": []
}


def save_webdataset(subset: str, features: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_roberta_path = DB_PROCESSED / 'xml_roberta' / 'meld_xml_roberta.pkl'
    with open(xml_roberta_path, 'rb') as f:
        xml_roberta_dict = pickle.load(f)

    subset_data = xml_roberta_dict[subset]
    subset_ids = sorted(list(subset_data.keys()))

    with wds.ShardWriter(str(output_dir / f"meld_{subset}_%06d.tar"), maxcount=1000) as sink:
        for sample_id in tqdm(subset_ids, total=len(subset_ids), desc=f"MELD {subset}"):

            if sample_id in IGNORE_DICT[subset]: continue

            sentiment = subset_data[sample_id]['sentiment']
            emotion_class = subset_data[sample_id]['emotion_class']
            xml_roberta_full = subset_data[sample_id]['xml_roberta'] # (120, 768)

            clip_path = DB_PROCESSED / 'clip' / subset / f'{sample_id}.npy'
            wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / subset / f'{sample_id}.pkl'

            if not clip_path.exists() or not wavlm_path.exists():
                raise ValueError(f"Missing features for sample_id: {sample_id}")

            clip_full = np.load(clip_path)  # (T_vid, F)
            with open(wavlm_path, 'rb') as f:
                wavlm_full = np.array(pickle.load(f))  # (12, T_aud, F)
                wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_aud, F)

            clip, clip_mask = pad_or_crop_time_dim(clip_full, CLIP_TIME_DIM)
            wavlm, wavlm_mask = pad_or_crop_time_dim(wavlm_full, WAVLM_BASEPLUS_TIME_DIM)
            xml_roberta, xml_roberta_mask = pad_or_crop_time_dim(xml_roberta_full, XML_ROBERTA_TIME_DIM)

            # Verify feature dimensions
            assert wavlm.ndim == 2 and wavlm.shape[1] == WAVLM_BASEPLUS_FEATURE_DIM, \
                f"Invalid WavLM shape: {wavlm.shape}"
            assert clip.ndim == 2 and clip.shape[1] == CLIP_FEATURE_DIM, \
                f"Invalid CLIP shape: {clip.shape}"
            assert xml_roberta.ndim == 2 and xml_roberta.shape[1] == XML_ROBERTA_FEATURE_DIM, \
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
                'sentiment_class.npy': np.array(sentiment, np.int64),
                'emotion_class.npy': np.array(emotion_class, np.int64),
            }

            # Add selected features
            sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
            sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

            sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['test']: #, 'valid', 'test']:
        save_webdataset(subset=subset, features=['clip', 'wavlm_baseplus', 'xml_roberta'], output_dir=output_dir)