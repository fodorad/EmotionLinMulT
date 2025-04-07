
if __name__ == "__main__":
    import time
    import numpy as np
    from tqdm import tqdm
    from exordium.video.facedetector import RetinaFaceDetector
    from exordium.video.tracker import IouTracker
    from exordium.video.fabnet import FabNetWrapper
    from exordium.video.opengraphau import OpenGraphAuWrapper
    from exordium.audio.smile import OpensmileWrapper
    from exordium.audio.clap import ClapWrapper
    from exordium.audio.io import load_audio_from_video

    parser = argparse.ArgumentParser(description="Script to preprocess MEAD.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start',  type=int, default=0, help='participant id slice start')
    parser.add_argument('--end',    type=int, default=50, help='participant id slice end')
    args = parser.parse_args()

    print(f"Using GPU ID: {args.gpu_id}")
    face_detector = RetinaFaceDetector(gpu_id=args.gpu_id, batch_size=10)
    fabnet_extractor = FabNetWrapper(gpu_id=args.gpu_id)
    au_extractor = OpenGraphAuWrapper(gpu_id=args.gpu_id)
    smile_lld = OpensmileWrapper(feature_level='lld')
    smile_f = OpensmileWrapper(feature_level='functionals')
    clap_extractor = ClapWrapper()

    participant_paths = sorted(list(DB.glob("*")))[args.start:args.end] # [::-1]
    print('Number of participants:', len(participant_paths))

    for p, participant_path in tqdm(enumerate(participant_paths), total=len(participant_paths), desc='Participants'):
        meadperson = MeadPerson(participant_path)

        for i, video_path in tqdm(enumerate(meadperson), total=len(meadperson), desc='Videos'):
            start_time = time.time()
            camera_position = MeadPerson.get_camera_position(video_path)
            label_emotion = MeadPerson.get_class(video_path)
            label_intensity = MeadPerson.get_intensity(video_path)
            label_emotion_name = MeadPerson.get_class_name(video_path)
            video_id = MeadPerson.get_video_id(video_path)
            sample_id = f'{camera_position}-{label_emotion}-{label_intensity}-{video_id}'
            print(meadperson)
            print(f"[{p+1}/{len(participant_paths)}-{i}/{len(meadperson)}] sample id:", sample_id)

            try:
                audio_path = participant_path / 'audio' / label_emotion_name / f'level_{label_intensity}' / f'{video_id}.m4a'
                audio_sample_id = f'{label_emotion}-{label_intensity}-{video_id}'

                clap_path = DB_PROCESSED / meadperson.id / 'clap' / f'{audio_sample_id}'
                clap_path.parent.mkdir(parents=True, exist_ok=True)
                if not (clap_path.parent / f'{clap_path.name}.npz').exists():
                    # audio_44100 = load_audio_from_video(video_path, sr=44100) # for clap (T,)
                    # segments_44100 = split_audio(audio_44100, 3, 44100)
                    # clap = np.squeeze(clap_extractor(audio_44100).cpu().detach().numpy()) # (1024,)
                    # clap_s = np.array([np.squeeze(clap_extractor(segment).cpu().detach().numpy()) for segment in segments_44100]) # (T_s, 1024)
                    # np.savez(str(clap_path), clap=clap, clap_segments=clap_s)
                    audio_44100 = load_m4a_to_numpy(str(audio_path), 44100)
                    clap = np.squeeze(clap_extractor(audio_44100).cpu().detach().numpy()) # (1024,)
                    if contains_nan(audio_44100):
                        with open('audio_44100.txt', 'a') as file:
                            file.write(f'{meadperson.id}-clap-{audio_sample_id} {str(audio_path)}\n')
                    if contains_nan(clap):
                        with open('clap.txt', 'a') as file:
                            file.write(f'{meadperson.id}-clap-{audio_sample_id} {str(audio_path)}\n')
                    np.save(str(clap_path), clap)
                    print('clap:', str(clap_path))

                egemaps_path = DB_PROCESSED / meadperson.id / 'egemaps' / f'{audio_sample_id}'
                egemaps_path.parent.mkdir(parents=True, exist_ok=True)
                if not (egemaps_path.parent / f'{egemaps_path.name}.npz').exists() or True:
                    #audio_16000 = load_audio_from_video(video_path, sr=16000) # for opensmile (T,)
                    #egemaps_lld = smile_lld(audio_16000) # (T, 25)
                    #egemaps_f = np.squeeze(smile_f(audio_16000)) # (88,)
                    #segments_16000 = split_audio(audio_16000, 3, 16000)
                    #egemaps_s_f = np.array([np.squeeze(smile_f(segment)) for segment in segments_16000]) # (T_s, 88)
                    #if contains_nan(egemaps_lld) or contains_nan(egemaps_f):
                    #    with open('egemaps.txt', 'a') as file:
                    #        file.write(f'{str(egemaps_path.parent / f"{egemaps_path.name}.npz")}\n')
                    #np.savez(str(egemaps_path), lld=egemaps_lld, functionals_full=egemaps_f, functionals_segments=egemaps_s_f)
                    audio_16000 = load_m4a_to_numpy(str(audio_path), 16000)
                    egemaps_lld = smile_lld(audio_16000) # (T, 25)
                    egemaps_f = np.squeeze(smile_f(audio_16000)) # (88,)
                    if contains_nan(audio_16000):
                        with open('audio_16000.txt', 'a') as file:
                            file.write(f'{meadperson.id}-egemaps_lld-{audio_sample_id} {str(audio_path)}\n')
                    if contains_nan(egemaps_lld):
                        with open('egemaps_lld.txt', 'a') as file:
                            file.write(f'{meadperson.id}-egemaps_lld-{audio_sample_id} {str(audio_path)}\n')
                    if contains_nan(egemaps_f):
                        with open('egemaps_f.txt', 'a') as file:
                            file.write(f'{meadperson.id}-egemaps_functionals_full-{audio_sample_id} {str(audio_path)}\n')
                    np.savez(str(egemaps_path), lld=egemaps_lld, functionals_full=egemaps_f)
                    print('egemaps:', str(egemaps_path))

            except Exception as e:
                #import socket
                #import traceback
                #hostname = socket.gethostname()
                with open(f"skip_uuids_audio.txt", "a") as f:
                    f.write(f'{meadperson.id}-{label_emotion}-{label_intensity}-{video_id}\n')
                    #traceback.TracebackException.from_exception(e).print(file=f)

            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time:03f} seconds")

            if (DB_PROCESSED / meadperson.id / 'tracker' / f'{sample_id}.vdet').exists() and \
               (DB_PROCESSED / meadperson.id / 'fabnet' / f'{sample_id}.pkl').exists() and \
               (DB_PROCESSED / meadperson.id / 'opengraphau' / f'{sample_id}.pkl').exists():
               continue

            try:
                videodetections = face_detector.detect_video(video_path, output_path=DB_PROCESSED / meadperson.id / 'tracker' / f'{sample_id}.vdet')
                track = IouTracker(max_lost=30).label(videodetections).merge().get_center_track()
                print("detected track length:", len(track))

                ids, embeddings = fabnet_extractor.track_to_feature(track, batch_size=30, output_path=DB_PROCESSED / meadperson.id / 'fabnet' / f'{sample_id}.pkl')
                print('fabnet embeddings:', embeddings.shape)

                #ids, landmarks = landmark_detector.track_to_feature(track, output_path=DB_PROCESSED / meadperson.id / 'facemesh' / f'{sample_id}.pkl')
                #print('facemesh landmarks:', landmarks.shape)

                ids, au = au_extractor.track_to_feature(track, batch_size=30, output_path=DB_PROCESSED / meadperson.id / 'opengraphau' / f'{sample_id}.pkl')
                print('au:', au.shape)
            except Exception as e:
                #import socket
                #import traceback
                #hostname = socket.gethostname()
                #with open(f"{hostname}_{meadperson.id}_{sample_id}.txt", "w") as f:
                #    traceback.TracebackException.from_exception(e).print(file=f)
                with open(f"skip_uuids_video.txt", "a") as f:
                    f.write(f'{meadperson.id}-{label_emotion}-{label_intensity}-{video_id}\n')

            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time:03f} seconds")