import argparse
import time
from pathlib import Path
from tqdm import tqdm
from exordium.video.clip import ClipWrapper

DB = Path('data/db/MEAD')
DB_PROCESSED = Path('data/db_processed/MEAD')

EMOTION_TO_CLASS = {
    'neutral': 0,
    'angry': 1,
    'contempt': 2,
    'disgusted': 3,
    'happy': 4,
    'fear': 5,
    'sad': 6,
    'surprised': 7
}

CAMERA_POSITIONS = [
    'down',
    'front',
    'top',
    'left_30',
    'left_60',
    'right_30',
    'right_60'
]

def parse_mead_path(path: Path) -> dict:
    return {
        "participant_id": path.parents[4].name,                  # e.g., "M01"
        "camera_position": path.parents[2].name,                 # e.g., "front"
        "emotion_class": EMOTION_TO_CLASS[path.parents[1].name], # e.g., 4 (happy)
        "emotion_intensity": int(path.parent.name[-1]),          # e.g., 2
        "video_id": path.stem                                    # e.g., "001"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess MEAD visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start',  type=int, default=0, help='participant id slice start')
    parser.add_argument('--end',    type=int, default=50, help='participant id slice end')
    args = parser.parse_args()

    participant_paths = sorted(list(DB.glob("*")))[args.start:args.end]
    print('Number of participants:', len(participant_paths))

    ignore_sample_ids = {
        'M013': ['left_30-3-1-004', 'right_30-3-1-001'],
        'M019': ['left_60-3-3-027', 'left_60-3-3-029', 'left_60-5-1-002', 'left_60-5-1-003', 'left_60-5-1-004', 'left_60-5-1-005', 'left_60-5-1-006', 'left_60-5-1-007', 'left_60-5-1-010', 'left_60-5-1-013', 'left_60-5-1-014', 'right_60-7-2-017'],
        'W014': ['down-1-1-017', 'front-1-1-017', 'left_30-1-1-017', 'left_60-1-1-017', 'right_30-1-1-017', 'right_60-1-1-017', 'top-1-1-017'],
        'W017': ['down-3-2-029']
    }

    print(f"Using GPU ID: {args.gpu_id}")
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    for participant_path in tqdm(participant_paths, total=len(participant_paths), desc='Participants'):
        participant_id = participant_path.name
        video_dir = participant_path / 'video'
        video_paths = sorted(list(video_dir.glob('**/*.mp4')))

        for video_path in tqdm(video_paths, total=len(video_paths), desc=f'{participant_id} Videos'):
            info = parse_mead_path(video_path)
            sample_id = f"{info['camera_position']}-{info['emotion_class']}-{info['emotion_intensity']}-{info['video_id']}"

            if participant_id in ignore_sample_ids and sample_id in ignore_sample_ids[participant_id]: continue # skip broken samples

            clip_path = DB_PROCESSED / participant_id / 'clip' / f'{sample_id}.npy'
            if not clip_path.exists():                 
                features = clip_extractor.extract_from_video(video_path, output_path=clip_path)