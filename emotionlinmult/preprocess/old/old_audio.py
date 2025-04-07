
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
