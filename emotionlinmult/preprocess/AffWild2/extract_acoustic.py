import os
import argparse
import random
import time
from tqdm import tqdm
from pathlib import Path
from exordium.audio.io import video2audio
from exordium.audio.wavlm import WavlmWrapper


DB = Path('data/db/Aff-Wild2')
DB_PROCESSED = Path('data/db_processed/Aff-Wild2')


def convert(video_path: Path):
    audio_path = DB_PROCESSED / 'audio' / f'{video_path.stem}.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    video2audio(video_path, audio_path, sr=16000)
    return audio_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess Aff-Wild2 acoustic features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=600, help='end index')
    args = parser.parse_args()

    [convert(elem) for elem in list((DB / 'videos').glob('**/*.mp4')) + list((DB / 'videos').glob('**/*.avi'))]
    audio_paths = sorted([elem for elem in list((DB_PROCESSED / 'audio').glob('**/*.wav'))])[args.start:args.end]
    random.shuffle(audio_paths)
    print(f"Found {len(audio_paths)} audios ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    wavlm_extractor = WavlmWrapper(gpu_id=args.gpu_id)

    for audio_path in tqdm(audio_paths, total=len(audio_paths), desc='Aff-Wild2 audios'):
        sample_id = Path(audio_path).stem

        wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl'
        if not wavlm_path.exists():
            features = wavlm_extractor.audio_to_feature(audio_path, output_path=wavlm_path)