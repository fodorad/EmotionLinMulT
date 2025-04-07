import argparse
import time
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from exordium.audio.io import video2audio
#from exordium.audio.smile import OpensmileWrapper
#from exordium.audio.clap import ClapWrapper
from emotionlinmult.preprocess.mead import *


def load_m4a_to_numpy(file_path: str, sr: int) -> np.ndarray:
    audio = AudioSegment.from_file(file_path, format='m4a')

    if audio.channels > 1:
        audio = audio.set_channels(1)  # Convert to mono

    if audio.frame_rate != sr:
        audio = audio.set_frame_rate(sr)

    samples = np.array(audio.get_array_of_samples())
    return samples.astype(np.float32) / np.max(np.abs(samples))  # Normalize to [-1, 1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to preprocess MEAD audio modality.")
    parser.add_argument('--start',  type=int, default=0, help='participant id slice start')
    parser.add_argument('--end',    type=int, default=50, help='participant id slice end')
    args = parser.parse_args()

    #smile_lld = OpensmileWrapper(feature_level='lld')
    #smile_f = OpensmileWrapper(feature_level='functionals')
    #clap_extractor = ClapWrapper()

    participant_paths = sorted(list(DB.glob("*")))[args.start:args.end]
    print('Number of participants:', len(participant_paths))

    for p, participant_path in tqdm(enumerate(participant_paths), total=len(participant_paths), desc='Participants'):
        meadperson = MeadPerson(participant_path)

        for i, video_path in tqdm(enumerate(meadperson), total=len(meadperson), desc='Videos'):
            start_time = time.time()
            sample_id = MeadPerson.get_sample_id(video_path)

            print(meadperson)
            print(f"[{p+1}/{len(participant_paths)}-{i}/{len(meadperson)}] sample id:", sample_id)

            audio_path = DB_PROCESSED / meadperson.id / 'audio_wav' / f'{sample_id}.wav'
            try:
                video2audio(video_path, audio_path, sr=44100, overwrite=False)
            except Exception as e:
                full_id = MeadPerson.get_full_id(video_path)
                with open(f"skip_uuids_audio.txt", "a") as f:
                    f.write(f'{full_id}\n')

            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time:03f} seconds")