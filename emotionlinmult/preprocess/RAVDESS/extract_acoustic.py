import argparse
from tqdm import tqdm
from pathlib import Path
from exordium.audio.wavlm import WavlmWrapper
from emotionlinmult.preprocess.RAVDESS import DB, DB_PROCESSED


if __name__ == "__main__":
    "source: https://zenodo.org/records/1188976"

    parser = argparse.ArgumentParser(description="Script to preprocess RAVDESS acoustic features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=5000, help='end index')
    args = parser.parse_args()

    video_paths = sorted(
        [elem for elem in list(DB.glob('**/*.mp4')) \
            if int(elem.stem.split("-")[0]) == 1 and \
               int(elem.stem.split("-")[1]) == 1] # 0: AV; 1: speech;
    )[args.start:args.end]
    audio_paths = [elem.parent / ("03-" + "-".join(elem.stem.split("-")[1:]) + ".wav") for elem in video_paths]
    random.shuffle(audio_paths)
    print(f"Processing {len(audio_paths)} audios ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    wavlm_extractor = WavlmWrapper(gpu_id=args.gpu_id)

    for audio_path in tqdm(audio_paths, total=len(audio_paths), desc='RAVDESS audios'):
        sample_id = "01-" + "-".join(Path(audio_path).stem.split('-')[1:])

        wavlm_baseplus_path = DB_PROCESSED / 'wavlm_baseplus' / f'{sample_id}.pkl'
        if not wavlm_baseplus_path.exists():
            features = wavlm_extractor.audio_to_feature(audio_path, output_path=wavlm_baseplus_path)