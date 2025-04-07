import os
import json
import cv2
import numpy as np
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from exordium.video.fabnet import FabNetWrapper
from exordium.video.opengraphau import OpenGraphAuWrapper
from exordium.video.clip import ClipWrapper
from exordium.video.bb import xyxy2xywh, xywh2midwh, crop_mid


DB = Path('data/db/AFEW-VA')
DB_PROCESSED = Path('data/db_processed/AFEW-VA')


def crop_face(frame_dir: str, annotation_file: str, output_dir: str, extra_space: float = 1.5):
    """
    Crop head based on landmarks and save cropped faces frame-wise.

    Args:
        frame_dir (str or Path): Path to the directory containing frames.
        annotation_file (str or Path): Path to the annotation JSON file.
        output_dir (str or Path): Path to the directory to save cropped faces.
    """
    # Ensure paths are Path objects
    frame_dir = Path(frame_dir)
    annotation_file = Path(annotation_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the annotation JSON
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    frames_info = annotations.get("frames", {})
    
    for frame_id, frame_data in frames_info.items():
        output_path = output_dir / f"{frame_id}.png"
        if output_path.exists(): continue

        # Read the frame
        frame_path = frame_dir / f"{frame_id}.png"
        if not frame_path.exists():
            print(f"Frame {frame_path} does not exist, skipping.")
            continue
        frame = cv2.imread(str(frame_path))

        # Get landmarks
        landmarks = frame_data.get("landmarks", [])
        if not landmarks:
            print(f"No landmarks for frame {frame_id}, skipping.")
            continue
        landmarks = np.array(landmarks)

        # Determine bounding box
        x_min, y_min = landmarks.min(axis=0).astype(int)
        x_max, y_max = landmarks.max(axis=0).astype(int)

        # Ensure bounding box is within frame dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        bb_xywh = xyxy2xywh([x_min, y_min, x_max, y_max])
        bb_midwh = xywh2midwh(bb_xywh)
        cropped_face = crop_mid(frame, bb_midwh[:2], np.rint(max(bb_xywh[2:]) * extra_space).astype(int))

        # Save the cropped face
        cv2.imwrite(str(output_path), cropped_face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess AFEW-VA visual features.")
    parser.add_argument('--gpu_id', type=int, default=-1, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=600, help='end index')
    args = parser.parse_args()

    video_dirs = sorted(list((DB / 'samples').glob("*")))[args.start:args.end][::-1]
    print(f"Found {len(video_dirs)} video dirs ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    fabnet_extractor = FabNetWrapper(gpu_id=args.gpu_id)
    au_extractor = OpenGraphAuWrapper(gpu_id=args.gpu_id)
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    for i, video_dir in tqdm(enumerate(video_dirs), total=len(video_dirs), desc='Videos'):
        start_time = time.time()
        sample_id = Path(video_dir).stem
        annotation_path = video_dir / f'{sample_id}.json'
        face_dir = DB_PROCESSED / 'face' / sample_id
        crop_face(video_dir, annotation_path, face_dir)
        face_paths = sorted(list(face_dir.glob('*.png')))

        if (DB_PROCESSED / 'fabnet' / f'{sample_id}.pkl').exists() and \
           (DB_PROCESSED / 'opengraphau' / f'{sample_id}.pkl').exists() and \
           (DB_PROCESSED / 'clip' / f'{sample_id}.pkl').exists():
            continue

        print(f'Sample id: {sample_id}')

        try:
            if not (DB_PROCESSED / 'clip' / f'{sample_id}.pkl').exists():
                frame_paths = sorted(list(video_dir.glob('*.png')))
                ids, features = clip_extractor.dir_to_feature(frame_paths, batch_size=15, verbose=True, output_path=DB_PROCESSED / 'clip' / f'{sample_id}.pkl')
                print('clip features:', features.shape)

            if not (DB_PROCESSED / 'fabnet' / f'{sample_id}.pkl').exists():
                ids, embeddings = fabnet_extractor.dir_to_feature(face_paths, batch_size=15, verbose=True, output_path=DB_PROCESSED / 'fabnet' / f'{sample_id}.pkl')
                print('fabnet embeddings:', embeddings.shape)

            if not (DB_PROCESSED / 'opengraphau' / f'{sample_id}.pkl').exists():
                ids, au = au_extractor.dir_to_feature(face_paths, batch_size=15, verbose=True, output_path=DB_PROCESSED / 'opengraphau' / f'{sample_id}.pkl')
                print('au:', au.shape)

        except Exception as e:
            with open(f"data/db_processed/AFEW-VA/skip_feat.txt", "a") as f:
                f.write(f"{video_dir}\t{sample_id}\t{str(e)}\n")
            print(e)
            pass

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:03f} seconds")