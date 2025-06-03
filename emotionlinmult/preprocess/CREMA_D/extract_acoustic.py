import os
import argparse
import random
import time
from tqdm import tqdm
from pathlib import Path
from exordium.audio.io import video2audio
from exordium.audio.smile import OpensmileWrapper
from exordium.audio.wavlm import WavlmWrapper


DB = Path('data/db/CREMA-D')
DB_PROCESSED = Path('data/db_processed/CREMA-D')


def convert(audio_orig: Path):
    audio_path = DB_PROCESSED / 'audio' / f'{audio_orig.stem}.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    video2audio(audio_orig, audio_path, sr=16000)
    return audio_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess CREMA-D acoustic features.")
    parser.add_argument('--gpu_id', type=int, default=-1, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=8000, help='end index')
    args = parser.parse_args()

    audio_orig = list((DB_PROCESSED / 'AudioWAV').glob('**/*.wav'))
    random.shuffle(audio_orig)
    for elem in tqdm(audio_orig, total=len(audio_orig), desc="Convert to 16k sr"):
        convert(elem)

    audio_paths = sorted([elem for elem in list((DB_PROCESSED / 'audio').glob('**/*.wav'))])[args.start:args.end]
    random.shuffle(audio_paths)
    print(f"Found {len(audio_paths)} audios ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    wavlm_extractor = WavlmWrapper(gpu_id=args.gpu_id)

    for audio_path in tqdm(audio_paths, total=len(audio_paths), desc='CelebV-HQ audios'):
        sample_id = Path(audio_path).stem

        wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl'
        if not wavlm_path.exists():
            features = wavlm_extractor.audio_to_feature(audio_path, output_path=wavlm_path)