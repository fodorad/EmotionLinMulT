import os
import random
import json
import cv2
import numpy as np
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from exordium.video.clip import ClipWrapper


DB = Path('data/db/VEATIC')
DB_PROCESSED = Path('data/db_processed/VEATIC')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess VEATIC visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=125, help='end index')
    args = parser.parse_args()

    video_paths = sorted(list((DB / 'videos').glob("*.mp4")))[args.start:args.end]
    random.shuffle(video_paths)

    print(f"Using GPU ID: {args.gpu_id}")
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    for video_path in tqdm(video_paths, total=len(video_paths), desc='VEATIC visual'):
        sample_id = Path(video_path).stem

        clip_path = DB_PROCESSED / 'clip' / f'{sample_id}.npy'
        if not clip_path.exists():
            features = clip_extractor.extract_from_video(video_path, output_path=clip_path)