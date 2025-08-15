import argparse
from decord import VideoReader
import cv2
from pathlib import Path


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mead_dir", type=str, default="data/db/MEAD")
    parser.add_argument("--participant_id", type=str, default="W036")
    parser.add_argument("--emotion", type=str, default="happy")
    parser.add_argument("--level", type=str, default="level_3")
    parser.add_argument("--clip_id", type=str, default="014")
    parser.add_argument("--frame_id", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="data/db_processed/MEAD/visualization/headpose")
    args = parser.parse_args()

    camera_ids = ['down', 'front', 'top', 'left_30', 'left_60', 'right_30', 'right_60']
    video_paths = [Path(args.mead_dir) / f"{args.participant_id}/video/{camera_id}/{args.emotion}/{args.level}/{args.clip_id}.mp4" for camera_id in camera_ids]
    for video_path in video_paths:
        vr = VideoReader(str(video_path))
        frame = vr[args.frame_id].asnumpy()
        output_dir = Path(args.output_dir) / args.participant_id
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.parents[2].name}.png"
        print(f'frame saved to: {output_path}')
        cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
