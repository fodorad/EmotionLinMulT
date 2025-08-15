from pathlib import Path
import numpy as np
from emotionlinmult.preprocess import CLIP_TIME_DIM
from emotionlinmult.preprocess.AFEWVA import DB

video_ids = sorted(list((DB / 'samples').glob("*")))
print(f"Found {len(video_ids)} videos.")

video_lengths = {}
for video_id in video_ids:
    n_frames = len(list(video_id.glob("*.png")))
    video_lengths[video_id.name] = n_frames

for video_id, n_frames in video_lengths.items():
    if n_frames > CLIP_TIME_DIM:
        print(f"{video_id}: {n_frames} frames > {CLIP_TIME_DIM}")

print('number of files:', len(video_lengths))
print('mean:', np.array(list(video_lengths.values())).mean())
print('min:', np.array(list(video_lengths.values())).min())
print('max:', np.array(list(video_lengths.values())).max())