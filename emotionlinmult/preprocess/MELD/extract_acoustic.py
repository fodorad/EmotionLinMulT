import os
import argparse
import random
import time
from tqdm import tqdm
from pathlib import Path
from exordium.audio.io import video2audio
from exordium.audio.wavlm import WavlmWrapper


DB = Path('data/db/MELD')
DB_PROCESSED = Path('data/db_processed/MELD')
SUBSET_NAME = {
    'train_splits': 'train',
    'dev_splits_complete': 'valid',
    'output_repeated_splits_test': 'test'
}


def convert(video_path: Path):
    audio_path = DB_PROCESSED / 'audio' / SUBSET_NAME[video_path.parent.name] / f'{video_path.stem}.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    video2audio(video_path, audio_path, sr=16000)
    return audio_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess MELD acoustic features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=15000, help='end index')
    args = parser.parse_args()

    video_paths = sorted([elem for elem in list(DB.glob('**/*.mp4')) if elem.stem[0] != '.'])[args.start:args.end]
    random.shuffle(video_paths)
    audio_paths = [convert(elem) for elem in video_paths]
    print(f"Found {len(audio_paths)} audios ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    wavlm_extractor = WavlmWrapper(gpu_id=args.gpu_id)

    IGNORE_DICT = {
        "train": ["dia125_utt3"],
        "valid": [],
        "test": []
    }

    for audio_path in tqdm(audio_paths, total=len(audio_paths), desc='MELD acoustic'): # 13848
        sample_id = Path(audio_path).stem
        subset = audio_path.parent.name

        if sample_id in IGNORE_DICT[subset]: continue

        wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / subset / f'{sample_id}.pkl'
        if not wavlm_path.exists():
            features = wavlm_extractor.audio_to_feature(audio_path, output_path=wavlm_path)