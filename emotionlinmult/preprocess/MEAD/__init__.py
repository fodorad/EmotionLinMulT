from pathlib import Path


DB = Path('data/db/MEAD')
DB_PROCESSED = Path('data/db_processed/MEAD')


def parse_mead_acoustic_path(path: Path) -> dict:
    d = {
        "participant_id": path.parents[3].name,                        # e.g., "M001"
        "emotion_class": MEAD_EMOTION_NAME2ORIG[path.parents[1].name], # e.g., 4 (happy) from the 8 emotion classes
        "emotion_intensity": int(path.parent.name[-1]),                # e.g., 2 from the 3 intensity classes
        "audio_id": path.stem                                          # e.g., "001"
    }
    assert len(d['participant_id']) == 4, f"Invalid participant_id: {d['participant_id']} in {path}"
    assert d['emotion_class'] in MEAD_EMOTION_NAME2ORIG.values(), f"Invalid emotion_class: {d['emotion_class']} in {path}"
    assert d['emotion_intensity'] in [1, 2, 3], f"Invalid emotion_intensity: {d['emotion_intensity']} in {path}"
    assert len(d['audio_id']) == 3, f"Invalid audio_id: {d['audio_id']} in {path}"
    return d


def parse_mead_visual_path(path: Path) -> dict:
    d = {
        "participant_id": path.parents[4].name,                        # e.g., "M001"
        "camera_position": path.parents[2].name,                       # e.g., "front"
        "emotion_class": MEAD_EMOTION_NAME2ORIG[path.parents[1].name], # e.g., 4 (happy) from the 8 emotion classes
        "emotion_intensity": int(path.parent.name[-1]),                # e.g., 2 from the 3 intensity classes
        "video_id": path.stem                                          # e.g., "001"
    }
    assert len(d['participant_id']) == 4, f"Invalid participant_id: {d['participant_id']} in {path}"
    assert d['camera_position'] in MEAD_CAMERA_ORIG2ID.keys(), f"Invalid camera_position: {d['camera_position']} in {path}"
    assert d['emotion_class'] in MEAD_EMOTION_NAME2ORIG.values(), f"Invalid emotion_class: {d['emotion_class']} in {path}"
    assert d['emotion_intensity'] in [1, 2, 3], f"Invalid emotion_intensity: {d['emotion_intensity']} in {path}"
    assert len(d['video_id']) == 3, f"Invalid video_id: {d['video_id']} in {path}"
    return d
