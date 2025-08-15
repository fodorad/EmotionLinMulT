import os
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from exordium.audio.wavlm import WavlmWrapper
from emotionlinmult.preprocess.MEAD import DB, DB_PROCESSED, parse_mead_acoustic_path


IGNORE_SAMPLE_IDS = {
    'M041': ['6-2-011', '6-2-012', '6-2-013', '6-2-014'],
    'W014': ['1-1-017']
}


def convert(input_path, output_path):
    if Path(output_path).exists(): return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -i {str(input_path)} {str(output_path)}'
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess MEAD acoustic features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start',  type=int, default=0, help='participant id slice start')
    parser.add_argument('--end',    type=int, default=50, help='participant id slice end')
    args = parser.parse_args()

    participant_paths = sorted(list(DB.glob("*")))[args.start:args.end]
    random.shuffle(participant_paths)
    print('Number of participants:', len(participant_paths))

    print(f"Using GPU ID: {args.gpu_id}")
    wavlm_extractor = WavlmWrapper(gpu_id=args.gpu_id)

    for participant_path in tqdm(participant_paths, total=len(participant_paths), desc='Participants'):
        participant_id = participant_path.name
        audio_dir = participant_path / 'audio'
        audio_paths = sorted(list(audio_dir.glob('**/*.m4a')))

        for old_audio_path in tqdm(audio_paths, total=len(audio_paths), desc=f'{participant_id} Audios'):
            info = parse_mead_acoustic_path(old_audio_path)
            sample_id = f"{info['emotion_class']}-{info['emotion_intensity']}-{info['audio_id']}"
            audio_path = DB_PROCESSED / participant_id / 'audio' / f'{sample_id}.wav'

            if participant_id in IGNORE_SAMPLE_IDS and sample_id in IGNORE_SAMPLE_IDS[participant_id]: continue # skip broken samples
            convert(old_audio_path, audio_path)

            wavlm_baseplus_path = DB_PROCESSED / participant_id / 'wavlm_baseplus' / f'{sample_id}.pkl'
            if not wavlm_baseplus_path.exists():
                features = wavlm_extractor.audio_to_feature(audio_path, output_path=wavlm_baseplus_path)