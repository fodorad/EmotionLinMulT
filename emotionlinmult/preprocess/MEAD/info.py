import numpy as np
import os
import json
from pathlib import Path
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from emotionlinmult.preprocess.MEAD import DB, DB_PROCESSED


def get_mp4_lengths(root_dir, json_path):
    filepaths = list(Path(root_dir).glob('**/*.mp4'))

    d = {}
    for filepath in tqdm(filepaths, total=len(filepaths), desc='Videos'):
        try:
            video = VideoFileClip(str(filepath))
            duration = video.duration  # duration in seconds
            video.close()
            d[str(filepath)] = duration
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)

    return d


if __name__ == "__main__":

    lengths_dict = get_mp4_lengths(DB, DB_PROCESSED / 'video_lengths.json')

    for filename, length in lengths_dict.items():
        if length > 10:
            print(f"{filename}: {length:.2f} seconds")

    dur = np.array(list(lengths_dict.values()))
    print('number of files:', len(dur))
    print('mean:', dur.mean())
    print('min:', dur.min())
    print('max:', dur.max())