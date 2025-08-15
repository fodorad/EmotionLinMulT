import os
import argparse
import random
import time
from tqdm import tqdm
from pathlib import Path
from exordium.video.clip import ClipWrapper
from emotionlinmult.preprocess.CREMAD import DB_PROCESSED


def convert(flv_path, output_dir: Path = DB_PROCESSED):
    output_mp4_path = DB_PROCESSED / 'videos' / f'{flv_path.stem}.mp4'

    if not output_mp4_path.exists():
        output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
        os.system(f'ffmpeg -i {flv_path} -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k {output_mp4_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess CREMA-D visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=8000, help='end index')
    args = parser.parse_args()

    flv_paths = sorted(list((DB_PROCESSED / 'VideoFlash').glob('*.flv')))
    [convert(elem) for elem in tqdm(flv_paths, total=len(flv_paths))]
    video_paths = sorted(list((DB_PROCESSED / 'videos').glob('*.mp4')))[args.start:args.end]
    random.shuffle(video_paths)
    print(f"Found {len(video_paths)} videos ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    for video_path in tqdm(video_paths, total=len(video_paths), desc='CREMA-D videos'):
        sample_id = Path(video_path).stem

        clip_path = DB_PROCESSED / 'clip' / f'{sample_id}.npy'
        if not clip_path.exists():
            features = clip_extractor.extract_from_video(video_path, output_path=clip_path)