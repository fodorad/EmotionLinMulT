import os
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from exordium.audio.io import video2audio
#from exordium.audio.clap import ClapWrapper
from exordium.audio.smile import OpensmileWrapper
from exordium.audio.wavlm import WavlmWrapper


DB = Path('data/db/RAVDESS')
DB_PROCESSED = Path('data/db_processed/RAVDESS')


def convert(video_path: Path):
    audio_path = DB_PROCESSED / 'audio' / f'{video_path.stem}.wav'
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    video2audio(video_path, audio_path, sr=16000)
    return audio_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess RAVDESS acoustic features.")
    parser.add_argument('--gpu_id', type=int, default=-1, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=5000, help='end index')
    args = parser.parse_args()

    '''
    audio_paths = sorted(
        [convert(elem) for elem in list(DB.glob('**/*.mp4')) \
            if int(elem.stem.split("-")[0]) != 2]
    )[args.start:args.end]#[::-1]
    '''
    audio_paths = sorted(
        [elem for elem in list(DB.glob('**/*.wav')) \
         if int(elem.stem.split("-")[0]) == 3]
    )[args.start:args.end][::-1]
    print(f"Found {len(audio_paths)} audios ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    smile_extractor = OpensmileWrapper()
    wavlm_extractor = WavlmWrapper(gpu_id=args.gpu_id)
    # clap_extractor = ClapWrapper()

    for i, audio_path in tqdm(enumerate(audio_paths), total=len(audio_paths), desc='RAVDESS audios'):
        start_time = time.time()
        sample_id = Path(audio_path).stem

        if (DB_PROCESSED / 'egemaps_lld' / f'{sample_id}.npy').exists() and \
           (DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl').exists():
            continue

        print(f'Sample id: {sample_id}')

        try:
            if not (DB_PROCESSED / 'egemaps_lld' / f'{sample_id}.npy').exists():
                features = smile_extractor.audio_to_feature(audio_path, output_path=DB_PROCESSED / 'egemaps_lld' / f'{sample_id}.npy')
                print('egemaps_lld features:', features.shape)

            if not (DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl').exists():
                features = wavlm_extractor.audio_to_feature(audio_path, output_path=DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl')
                print('wavlm_baseplus last features:', features[-1].shape)

        except Exception as e:
            with open(f"data/db_processed/RAVDESS/skip_acoustic_feat.txt", "a") as f:
                f.write(f"{audio_path}\t{str(e)}\n")
            print(e)
            pass

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:03f} seconds")