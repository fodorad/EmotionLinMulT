import argparse
import time
from pathlib import Path
from tqdm import tqdm
from exordium.video.facedetector import RetinaFaceDetector
from exordium.video.tracker import IouTracker
from exordium.video.fabnet import FabNetWrapper
from exordium.video.opengraphau import OpenGraphAuWrapper
from exordium.video.clip import ClipWrapper

DB = Path('data/db/MEAD')
DB_PROCESSED = Path('data/db_processed/MEAD')

EMOTION_TO_ID = {
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
        "participant_id": path.parents[4].name,               # e.g., "M01"
        "camera_position": path.parents[2].name,              # e.g., "front"
        "emotion_class": EMOTION_TO_ID[path.parents[1].name], # e.g., 4 (happy)
        "emotion_intensity": int(path.parent.name[-1]),       # e.g., 2
        "video_id": path.stem                                 # e.g., "001"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess MEAD visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start',  type=int, default=0, help='participant id slice start')
    parser.add_argument('--end',    type=int, default=50, help='participant id slice end')
    args = parser.parse_args()

    print(f"Using GPU ID: {args.gpu_id}")
    face_detector = RetinaFaceDetector(gpu_id=args.gpu_id, batch_size=10)
    fabnet_extractor = FabNetWrapper(gpu_id=args.gpu_id)
    au_extractor = OpenGraphAuWrapper(gpu_id=args.gpu_id)
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    participant_paths = sorted(list(DB.glob("*")))[args.start:args.end]#[::-1]
    print('Number of participants:', len(participant_paths))

    for p, participant_path in tqdm(enumerate(participant_paths), total=len(participant_paths), desc='Participants'):
        participant_id = participant_path.name
        video_dir = participant_path / 'video'
        video_paths = sorted(list(video_dir.glob('**/*.mp4')))#[::-1]

        for i, video_path in tqdm(enumerate(video_paths), total=len(video_paths), desc=f'{args.start + p}:{participant_path} Videos'):
            start_time = time.time()
            info = parse_mead_path(video_path)
            sample_id = f"{info['camera_position']}-{info['emotion_class']}-{info['emotion_intensity']}-{info['video_id']}"

            try:
                if not (DB_PROCESSED / participant_id / 'clip' / f'{sample_id}.npy').exists():                 
                    features = clip_extractor.extract_from_video(video_path, output_path=DB_PROCESSED / participant_id / 'clip' / f'{sample_id}.npy')
                    print('clip features:', features.shape)

                if (DB_PROCESSED / participant_id / 'fabnet_wide' / f'{sample_id}.pkl').exists() and \
                   (DB_PROCESSED / participant_id / 'opengraphau_wide' / f'{sample_id}.pkl').exists():
                    continue

                videodetections = face_detector.detect_video(video_path, output_path=DB_PROCESSED / participant_id / 'tracker' / f'{sample_id}.vdet')
                track = IouTracker(max_lost=90).label(videodetections).merge().get_center_track()
                print("detected track length:", len(track))

                if not (DB_PROCESSED / participant_id / 'fabnet_wide' / f'{sample_id}.pkl').exists():
                    ids, embeddings = fabnet_extractor.track_to_feature(track, batch_size=30, output_path=DB_PROCESSED / participant_id / 'fabnet_wide' / f'{sample_id}.pkl')
                    print('fabnet embeddings:', embeddings.shape)

                if not (DB_PROCESSED / participant_id / 'opengraphau_wide' / f'{sample_id}.pkl').exists():
                    ids, au = au_extractor.track_to_feature(track, batch_size=30, output_path=DB_PROCESSED / participant_id / 'opengraphau_wide' / f'{sample_id}.pkl')
                    print('au:', au.shape)
        
            except Exception as e:
                with open(f"data/db_processed/MEAD/skip_uuids_video.txt", "a") as f:
                    f.write(f"{participant_id} {sample_id}.mp4\t{str(e)}\n")
                    print(e)

            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time:03f} seconds")