import os
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from exordium.video.tracker import IouTracker
from exordium.video.facedetector import RetinaFaceDetector
from exordium.video.fabnet import FabNetWrapper
from exordium.video.opengraphau import OpenGraphAuWrapper
from exordium.video.clip import ClipWrapper


DB = Path('data/db/RAVDESS')
DB_PROCESSED = Path('data/db_processed/RAVDESS')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess RAVDESS visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=5000, help='end index')
    args = parser.parse_args()

    video_paths = sorted(
        [elem for elem in list(DB.glob('**/*.mp4')) \
            if int(elem.stem.split("-")[0]) != 3] # and \
               #int(elem.stem.split("-")[1]) != 2] # these cases are the audio-only and song files, skip. 
    )[args.start:args.end][::-1]
    print(f"Found {len(video_paths)} videos ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    face_detector = RetinaFaceDetector(gpu_id=args.gpu_id, batch_size=10)
    fabnet_extractor = FabNetWrapper(gpu_id=args.gpu_id)
    au_extractor = OpenGraphAuWrapper(gpu_id=args.gpu_id)
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    for i, video_path in tqdm(enumerate(video_paths), total=len(video_paths), desc='RAVDESS videos'):
        start_time = time.time()
        sample_id = Path(video_path).stem

        if (DB_PROCESSED / 'fabnet' / f'{sample_id}.pkl').exists() and \
           (DB_PROCESSED / 'opengraphau' / f'{sample_id}.pkl').exists() and \
           (DB_PROCESSED / 'clip' / f'{sample_id}.npy').exists():
            continue

        print(f'Sample id: {sample_id}')

        try:
            videodetections = face_detector.detect_video(video_path, output_path=DB_PROCESSED / 'tracker' / f'{sample_id}.vdet')
            track = IouTracker(max_lost=30).label(videodetections).merge().get_center_track()
            print("detected track length:", len(track))

            if not (DB_PROCESSED / 'clip' / f'{sample_id}.npy').exists():
                features = clip_extractor.extract_from_video(video_path, verbose=True, output_path=DB_PROCESSED / 'clip' / f'{sample_id}.npy')
                print('clip features:', features.shape)

            if not (DB_PROCESSED / 'fabnet' / f'{sample_id}.pkl').exists():
                ids, embeddings = fabnet_extractor.dir_to_feature(face_paths, batch_size=15, verbose=True, output_path=DB_PROCESSED / 'fabnet' / f'{sample_id}.pkl')
                print('fabnet embeddings:', embeddings.shape)

            if not (DB_PROCESSED / 'opengraphau' / f'{sample_id}.pkl').exists():
                ids, au = au_extractor.dir_to_feature(face_paths, batch_size=15, verbose=True, output_path=DB_PROCESSED / 'opengraphau' / f'{sample_id}.pkl')
                print('au:', au.shape)

        except Exception as e:
            with open(f"data/db_processed/RAVDESS/skip_feat.txt", "a") as f:
                f.write(f"{video_path}\t{str(e)}\n")
            print(e)
            pass

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:03f} seconds")