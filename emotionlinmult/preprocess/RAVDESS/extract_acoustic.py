import os
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from exordium.audio.io import video2audio
from exordium.audio.wavlm import WavlmWrapper


DB = Path('data/db/RAVDESS')
DB_PROCESSED = Path('data/db_processed/RAVDESS')


def convert(video_path: Path):
    audio_path = DB_PROCESSED / 'audio' / f'{video_path.stem}.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    video2audio(video_path, audio_path, sr=16000)
    return audio_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess RAVDESS acoustic features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=5000, help='end index')
    args = parser.parse_args()

    audio_only_paths = sorted([elem for elem in list(DB.glob('**/*.wav'))])[args.start:args.end]
    audio_visual_paths = sorted([convert(elem) for elem in list(DB.glob('**/*.mp4')) if int(elem.stem.split("-")[0]) != 2])[args.start:args.end]
    audio_paths = audio_only_paths + audio_visual_paths
    print(f"Processing {len(audio_paths)} audios ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    wavlm_extractor = WavlmWrapper(gpu_id=args.gpu_id)

    for audio_path in tqdm(audio_paths, total=len(audio_paths), desc='RAVDESS audios'):
        sample_id = Path(audio_path).stem

        wavlm_baseplus_path = DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl'
        if not wavlm_baseplus_path.exists():
            features = wavlm_extractor.audio_to_feature(audio_path, output_path=wavlm_baseplus_path)
