import os
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from exordium.audio.io import video2audio
from exordium.audio.smile import OpensmileWrapper
from exordium.audio.wavlm import WavlmWrapper


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


def convert(m4a_path, output_path):
    if Path(output_path).exists(): return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -i {str(m4a_path)} {str(output_path)}'
    os.system(cmd)


def parse_mead_path(path: Path) -> dict:
    return {
        "participant_id": path.parents[3].name,               # e.g., "M01"
        "emotion_class": EMOTION_TO_ID[path.parents[1].name], # e.g., 4 (happy)
        "emotion_intensity": int(path.parent.name[-1]),       # e.g., 2
        "audio_id": path.stem                                 # e.g., "001"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess MEAD visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start',  type=int, default=0, help='participant id slice start')
    parser.add_argument('--end',    type=int, default=50, help='participant id slice end')
    args = parser.parse_args()

    participant_paths = sorted(list(DB.glob("*")))[args.start:args.end]#[::-1]
    print('Number of participants:', len(participant_paths))

    print(f"Using GPU ID: {args.gpu_id}")
    smile_extractor = OpensmileWrapper()
    wavlm_extractor = WavlmWrapper(gpu_id=args.gpu_id)

    for p, participant_path in tqdm(enumerate(participant_paths), total=len(participant_paths), desc='Participants'):
        participant_id = participant_path.name
        audio_dir = participant_path / 'audio'
        audio_paths = sorted(list(audio_dir.glob('**/*.m4a')))#[::-1]

        for i, old_audio_path in tqdm(enumerate(audio_paths), total=len(audio_paths), desc=f'{args.start + p}:{participant_path} Audios'):
            start_time = time.time()
            info = parse_mead_path(old_audio_path)
            sample_id = f"{info['emotion_class']}-{info['emotion_intensity']}-{info['audio_id']}"
            audio_path = DB_PROCESSED / participant_id / 'audio' / f'{sample_id}.wav'
            convert(old_audio_path, audio_path)

            start_time = time.time()

            if (DB_PROCESSED / participant_id / 'egemaps_lld' / f'{sample_id}.npy').exists() and \
               (DB_PROCESSED / participant_id / 'wavlm_baseplus' / f'{sample_id}.pkl').exists():
                continue

            print(f'Sample id: {sample_id}')

            try:
                if not (DB_PROCESSED / participant_id / 'egemaps_lld' / f'{sample_id}.npy').exists():
                    features = smile_extractor.audio_to_feature(audio_path, output_path=DB_PROCESSED / participant_id / 'egemaps_lld' / f'{sample_id}.npy')
                    print('egemaps_lld features:', features.shape)

                if not (DB_PROCESSED / participant_id / 'wavlm_baseplus' / f'{sample_id}.pkl').exists():
                    features = wavlm_extractor.audio_to_feature(audio_path, output_path=DB_PROCESSED / participant_id / 'wavlm_baseplus' / f'{sample_id}.pkl')
                    print('wavlm_baseplus last features:', features[-1].shape)

            except Exception as e:
                with open(f"data/db_processed/MEAD/skip_acoustic_feat.txt", "a") as f:
                    f.write(f"{audio_path}\t{str(e)}\n")
                print(e)
                pass

            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time:03f} seconds")