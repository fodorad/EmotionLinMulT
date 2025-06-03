from pathlib import Path
from tqdm import tqdm
import webdataset as wds
import numpy as np
import cv2
import json
import pickle
from exordium.utils.padding import pad_or_crop_time_dim
from sklearn.model_selection import train_test_split

DB = Path("data/db/CelebV-HQ")
DB_PROCESSED = Path("data/db_processed/CelebV-HQ")
ANNOTATION_PATH = Path("data/db/CelebV-HQ/celebvhq_info.json")


EMOTION_TO_CLASS = {
    'neutral': 0,
    'anger': 1,
    'contempt': 2,
    'disgust': 3,
    'happy': 4,
    'fear': 5,
    'sadness': 6,
    'surprise': 7
}
CLASS_TO_EMOTION = {v: k for k, v in EMOTION_TO_CLASS.items()}

APPEARANCE_MAPPING = ["blurry", "male", "young", "chubby", "pale_skin", "rosy_cheeks", "oval_face", "receding_hairline", "bald", "bangs", "black_hair", "blonde_hair", "gray_hair", "brown_hair", "straight_hair", "wavy_hair", "long_hair", "arched_eyebrows", "bushy_eyebrows", "bags_under_eyes", "eyeglasses", "sunglasses", "narrow_eyes", "big_nose", "pointy_nose", "high_cheekbones", "big_lips", "double_chin", "no_beard", "5_o_clock_shadow", "goatee", "mustache", "sideburns", "heavy_makeup", "wearing_earrings", "wearing_hat", "wearing_lipstick", "wearing_necklace", "wearing_necktie", "wearing_mask"]
ACTION_MAPPING = ["blow", "chew", "close_eyes", "cough", "cry", "drink", "eat", "frown", "gaze", "glare", "head_wagging", "kiss", "laugh", "listen_to_music", "look_around", "make_a_face", "nod", "play_instrument", "read", "shake_head", "shout", "sigh", "sing", "sleep", "smile", "smoke", "sneer", "sneeze", "sniff", "talk", "turn", "weep", "whisper", "wink", "yawn"] 


def get_appearance_names(labels: list[int]) -> list[str]:
    return [label for label, val in zip(APPEARANCE_MAPPING, labels) if val == 1]


def get_action_names(labels: list[int]) -> list[str]:
    return [label for label, val in zip(ACTION_MAPPING, labels) if val == 1]


def get_fps(video_path):
    """Get the frames per second (fps) of a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_all_fps():
    video_paths = sorted(list((DB / '35666').glob('*.mp4')))
    cache_path = DB_PROCESSED / 'fps.pkl'
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            fps_dict = pickle.load(f)
    else:
        fps_dict = {}
        for video_path in tqdm(video_paths, desc='FPS caching...'):
            video_id = video_path.stem
            fps = get_fps(video_path)
            fps_dict[video_id] = fps
        with open(cache_path, 'wb') as f:
            pickle.dump(fps_dict, f)
    return fps_dict


def load_annotations():
    with open(ANNOTATION_PATH) as f:
        data = json.load(f)
    return data['meta_info'], data['clips']


def process_emotion_intervals(clip_id, clip_data):
    """Split clips with multiple emotions into subclips"""
    emotion_info = clip_data['attributes']['emotion']
    appearance_info = clip_data['attributes']['appearance']
    action_info = clip_data['attributes']['action']
    samples = []

    if not emotion_info['sep_flag']:
        # Single emotion for entire clip
        samples.append({
            'clip_id': clip_id,
            'start_sec': 0,
            'end_sec': clip_data['duration']['end_sec'] - clip_data['duration']['start_sec'],
            'emotion': emotion_info['labels'],
            'appearance': np.array(appearance_info, np.int64),
            'action': np.array(action_info, np.int64)
        })
    else:
        # Multiple emotions - create subclips
        for ind, emotion_segment in enumerate(emotion_info['labels']):
            samples.append({
                'clip_id': f"{clip_id}_{ind:03d}",
                'start_sec': emotion_segment['start_sec'],
                'end_sec': emotion_segment['end_sec'],
                'emotion': emotion_segment['emotion'],
                'appearance': np.array(appearance_info, np.int64),
                'action': np.array(action_info, np.int64)
            })

    return samples


def save_webdataset(subset: str, features: list[str], output_dir: Path, analysis_window_sec: int = 10) -> None:
    meta_info, all_clips = load_annotations()
    fps_dict = get_all_fps()

    output_dir.mkdir(parents=True, exist_ok=True)

    ids = list(set([v['ytb_id'] for k, v in all_clips.items()])) # 13844
    train_ids, valid_test_ids = train_test_split(ids, test_size=0.3, random_state=42)
    valid_ids, test_ids = train_test_split(valid_test_ids, test_size=0.5, random_state=42)
    SUBSET_IDS = {
        'train': train_ids,
        'valid': valid_ids,
        'test': test_ids
    }
    subset_clips = {k: v for k, v in all_clips.items() if v['ytb_id'] in SUBSET_IDS[subset]}

    # 50 Hz for audio features
    audio_sr = 50 # before wavlm extraction, we downsample to 16k sr, and the output sr is 50
    audio_time_dim = analysis_window_sec * audio_sr
    clip_target_time_dim = 30 * analysis_window_sec # shared expected fps * window size

    with wds.ShardWriter(str(output_dir / f"celebvhq_{subset}_%06d.tar"), maxcount=1000) as sink:
        for clip_id, clip_data in tqdm(subset_clips.items(), desc=f"Processing {subset}"):

            video_fps = fps_dict[clip_id]
            clip_time_dim = int(np.rint(analysis_window_sec * video_fps))

            # Load visual features
            clip_path = DB_PROCESSED / 'clip' / f"{clip_id}.npy"
            clip_full = np.load(clip_path) if clip_path.exists() else None  # (T_vid, F)

            # Load audio features
            wavlm_path = DB_PROCESSED / 'wavlm_baseplus' / f"{clip_id}.pkl"
            if wavlm_path.exists():
                with open(wavlm_path, 'rb') as f:
                    wavlm_full = np.array(pickle.load(f))  # (12, T_aud, F)
                    wavlm_full = np.mean(wavlm_full[8:, :, :], axis=0)  # (T_aud, F)
            else:
                wavlm_full = None

            if clip_full is None and wavlm_full is None:
                raise ValueError(f"No features found for {clip_id}.")

            # Process emotion annotations and split if needed
            sample_dicts = process_emotion_intervals(clip_id, clip_data)

            for sample_dict in sample_dicts:
                # Calculate feature indices based on time
                duration = sample_dict['end_sec'] - sample_dict['start_sec']

                # Calculate frame/sample indices for this emotion segment
                vid_start= int(sample_dict['start_sec'] * video_fps)
                vid_end = int(sample_dict['end_sec'] * video_fps)
                aud_start = int(sample_dict['start_sec'] * audio_sr)
                aud_end = int(sample_dict['end_sec'] * audio_sr)

                # Determine chunking
                num_visual_frames = vid_end - vid_start if clip_full is not None else 0
                num_audio_frames = aud_end - aud_start if wavlm_full is not None else 0

                num_visual_chunks = (num_visual_frames + clip_target_time_dim - 1) // clip_time_dim if num_visual_frames else 0
                num_acoustic_chunks = (num_audio_frames + audio_time_dim - 1) // audio_time_dim if num_audio_frames else 0
                num_chunks = max(1, num_visual_chunks, num_acoustic_chunks)

                for chunk_idx in range(num_chunks):
                    chunk_key = f"{sample_dict['clip_id']}_{chunk_idx:05d}"

                    # Video chunk
                    if clip_full is not None:
                        v_start = vid_start + chunk_idx * clip_time_dim
                        v_end = vid_start + (chunk_idx + 1) * clip_time_dim
                        clip_chunk = clip_full[v_start:v_end]
                        clip, clip_mask = pad_or_crop_time_dim(clip_chunk, clip_target_time_dim)
                    else:
                        clip = np.zeros((clip_target_time_dim, 1024), np.float32)
                        clip_mask = np.zeros(clip_target_time_dim, bool)

                    # Audio chunk
                    if wavlm_full is not None:
                        a_start = aud_start + chunk_idx * audio_time_dim
                        a_end = aud_start + (chunk_idx + 1) * audio_time_dim
                        wavlm_chunk = wavlm_full[a_start:a_end]
                        wavlm, wavlm_mask = pad_or_crop_time_dim(wavlm_chunk, audio_time_dim)
                    else:
                        wavlm = np.zeros((audio_time_dim, 768), np.float32)
                        wavlm_mask = np.zeros(audio_time_dim, bool)

                    # Verify feature dimensions
                    assert wavlm.ndim == 2 and wavlm.shape[1] == 768, \
                        f"Invalid WavLM shape: {wavlm.shape}"
                    assert clip.ndim == 2 and clip.shape[1] == 1024, \
                        f"Invalid CLIP shape: {clip.shape}"

                    # Build feature dictionary
                    feat_dict = {
                        'clip': clip,
                        'clip_mask': clip_mask,
                        'wavlm_baseplus': wavlm,
                        'wavlm_baseplus_mask': wavlm_mask,
                    }

                    # Create sample
                    sample = {
                        '__key__': chunk_key,
                        'emotion_class.npy': np.array(EMOTION_TO_CLASS[sample_dict['emotion']], np.int64),
                        'appearance.npy': np.array(sample_dict['appearance'], np.int64),
                        'action.npy': np.array(sample_dict['action'], np.int64),
                    }

                    # Add selected features
                    sample.update({f"{feature}.npy": feat_dict[feature] for feature in features if feature in feat_dict})
                    sample.update({f"{feature}_mask.npy": feat_dict[f"{feature}_mask"] for feature in features if f"{feature}_mask" in feat_dict})

                    sink.write(sample)


if __name__ == "__main__":
    output_dir = DB_PROCESSED / "webdataset"
    for subset in ['valid', 'test']: # 'train'
        save_webdataset(subset, ['clip', 'wavlm_baseplus'], output_dir)
