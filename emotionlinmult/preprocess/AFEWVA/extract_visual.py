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
from exordium.video.bb import xyxy2xywh, xywh2midwh, crop_mid
from emotionlinmult.preprocess.AFEWVA import DB, DB_PROCESSED


def crop_face(frame_dir: str, annotation_file: str, output_dir: str, extra_space: float = 1.5):
    """Crop head based on landmarks and save cropped faces frame-wise.

    Args:
        frame_dir (str or Path): Path to the directory containing frames.
        annotation_file (str or Path): Path to the annotation JSON file.
        output_dir (str or Path): Path to the directory to save cropped faces.
    """
    frame_dir = Path(frame_dir)
    annotation_file = Path(annotation_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    frames_info = annotations.get("frames")
    
    for frame_id, frame_data in frames_info.items():
        output_path = output_dir / f"{frame_id}.png"
        if output_path.exists(): continue

        # Read the frame
        frame_path = frame_dir / f"{frame_id}.png"
        frame = cv2.imread(str(frame_path))

        # Get landmarks
        landmarks = frame_data.get("landmarks")
        landmarks = np.array(landmarks)

        # Determine bounding box
        x_min, y_min = landmarks.min(axis=0).astype(int)
        x_max, y_max = landmarks.max(axis=0).astype(int)

        # Ensure bounding box is within frame dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        # Crop the face
        bb_xywh = xyxy2xywh([x_min, y_min, x_max, y_max])
        bb_midwh = xywh2midwh(bb_xywh)
        cropped_face = crop_mid(frame, bb_midwh[:2], np.rint(max(bb_xywh[2:]) * extra_space).astype(int))
        cv2.imwrite(str(output_path), cropped_face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess AFEW-VA visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=600, help='end index')
    args = parser.parse_args()

    video_dirs = sorted(list((DB / 'samples').glob("*")))[args.start:args.end]
    random.shuffle(video_dirs)

    print(f"Using GPU ID: {args.gpu_id}")
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    for video_dir in tqdm(video_dirs, total=len(video_dirs), desc='AFEW-VA visual'):
        sample_id = Path(video_dir).stem
        annotation_path = video_dir / f'{sample_id}.json'
        face_dir = DB_PROCESSED / 'face' / sample_id
        crop_face(video_dir, annotation_path, face_dir)

        clip_path = DB_PROCESSED / 'clip' / f'{sample_id}.pkl'
        if not clip_path.exists():
            frame_paths = sorted(list(video_dir.glob('*.png')))
            ids, features = clip_extractor.dir_to_feature(frame_paths, batch_size=15, verbose=False, output_path=clip_path)

        clip_face_path = DB_PROCESSED / 'clip_face' / f'{sample_id}.pkl'
        if not clip_face_path.exists():
            face_paths = sorted(list(face_dir.glob('*.png')))
            ids, features = clip_extractor.dir_to_feature(face_paths, batch_size=15, verbose=False, output_path=clip_face_path)