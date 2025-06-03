import os
import argparse
from enum import Enum
import pickle
import random
import time
from tqdm import tqdm
from pathlib import Path
from exordium.text.whisper import WhisperWrapper
from exordium.text.xml_roberta import XmlRobertaWrapper
from emotionlinmult.preprocess.CAER.create_webdataset import EMOTION_TO_CLASS


DB = Path('data/db/CAER')
DB_PROCESSED = Path('data/db_processed/CAER')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess CAER acoustic features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=14000, help='end index')
    args = parser.parse_args()

    audio_paths = sorted([elem for elem in list((DB_PROCESSED / 'audio').glob('**/*.wav'))])[args.start:args.end]
    random.shuffle(audio_paths)
    print(f"Found {len(audio_paths)} audios ({args.start}-{args.end})...")

    print(f"Using CPU")
    whisper = WhisperWrapper()
    xml_roberta = XmlRobertaWrapper()

    samples_whisper = {}
    samples = {}
    for audio_path in tqdm(audio_paths, total=len(audio_paths), desc='CAER whisper & xml roberta'):
        start_time = time.time()
        subset = Path(audio_path).parent.parent.name
        label = Path(audio_path).parent.name
        clip_id = Path(audio_path).stem
        sample_id = f'{subset}_{label}_{clip_id}'

        output = whisper(str(audio_path))
        samples_whisper[sample_id] = output # dict or None
        text = output['text'] if len(output['text']) > 0 else None

        if text is not None:
            token_embeddings, _ = xml_roberta(text)
            token_embeddings = token_embeddings.detach().cpu().numpy().squeeze(0) # (1, T, F) -> (T, F)
            assert token_embeddings.ndim == 2 and token_embeddings.shape[0] != 0 and token_embeddings.shape[1] == 768
        else:
            token_embeddings = None

        samples[sample_id] = {
            'xml_roberta': token_embeddings, # (T, F) or None
            'text': input_featurestext, # str or None
            'emotion_class': EMOTION_TO_CLASS[label.lower()], # str
        }

    output_path = DB_PROCESSED / 'xml_roberta' / f'caer_xml_roberta.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    output_path = DB_PROCESSED / 'whisper' / f'caer_whisper.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(samples_whisper, f)