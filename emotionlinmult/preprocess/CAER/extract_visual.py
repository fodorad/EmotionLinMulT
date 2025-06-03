import os
import argparse
import random
import time
from tqdm import tqdm
from pathlib import Path
from exordium.video.clip import ClipWrapper


DB = Path('data/db/CAER')
DB_PROCESSED = Path('data/db_processed/CAER')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess CAER visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=15000, help='end index')
    args = parser.parse_args()

    video_paths = sorted([elem for elem in list(DB.glob('**/*.avi'))])[args.start:args.end]
    random.shuffle(video_paths)
    print(f"Found {len(video_paths)} videos ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    for video_path in tqdm(video_paths, total=len(video_paths), desc='CAER videos'):
        subset = Path(video_path).parent.parent.name
        label = Path(video_path).parent.name
        sample_id = Path(video_path).stem

        clip_path = DB_PROCESSED / 'clip' / subset / label / f'{sample_id}.npy'
        if not clip_path.exists():
            features = clip_extractor.extract_from_video(video_path, verbose=True, output_path=clip_path)