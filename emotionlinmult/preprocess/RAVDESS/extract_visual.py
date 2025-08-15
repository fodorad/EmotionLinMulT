import argparse
import random
from tqdm import tqdm
from pathlib import Path
from exordium.video.clip import ClipWrapper
from emotionlinmult.preprocess.RAVDESS import DB, DB_PROCESSED


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess RAVDESS visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=5000, help='end index')
    args = parser.parse_args()

    video_paths = sorted(
        [elem for elem in list(DB.glob('**/*.mp4')) \
            if int(elem.stem.split("-")[0]) == 1 and \
               int(elem.stem.split("-")[1]) == 1] # 0: AV; 1: speech;
    )[args.start:args.end]
    random.shuffle(video_paths)
    print(f"Processing {len(video_paths)} videos ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    for video_path in tqdm(video_paths, total=len(video_paths), desc='RAVDESS videos'):
        sample_id = Path(video_path).stem

        clip_path = DB_PROCESSED / 'clip' / f'{sample_id}.npy'
        if not clip_path.exists():
            features = clip_extractor.extract_from_video(video_path, output_path=clip_path)