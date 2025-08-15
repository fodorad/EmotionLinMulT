from dataclasses import dataclass
from typing import Dict, Union, Optional, List
import numpy as np
import warnings


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".with_length\(\) only sets the value of __len__ for compatibility.*"
)


FPS = 30
SR = 50
ANALYSIS_WINDOW_SEC = 10
CLIP_FEATURE_DIM = 1024
WAVLM_BASEPLUS_FEATURE_DIM = 768
XML_ROBERTA_FEATURE_DIM = 768
CLIP_TIME_DIM = FPS * ANALYSIS_WINDOW_SEC  # 10 sec @ 30 fps = 300
WAVLM_BASEPLUS_TIME_DIM = SR * ANALYSIS_WINDOW_SEC  # 10 sec @ ~50 sr = 500
XML_ROBERTA_TIME_DIM = 120 # preliminary experiments

#################
# EMOTION_CLASS #
#################

@dataclass(frozen=True)
class UnifiedEmotionClass:
    unified_id: int
    unified_name: str


UNIFIED_EMOTION_CLASSES: List[UnifiedEmotionClass] = [
    UnifiedEmotionClass(0, "neutral"),
    UnifiedEmotionClass(1, "happy"),
    UnifiedEmotionClass(2, "surprise"),
    UnifiedEmotionClass(3, "sad"),
    UnifiedEmotionClass(4, "anger"),
    UnifiedEmotionClass(5, "fear"),
    UnifiedEmotionClass(6, "disgust"),
    UnifiedEmotionClass(7, "contempt")
]


def get_unified_emotion_class_name(unified_id: int) -> str:
    for uc in UNIFIED_EMOTION_CLASSES:
        if uc.unified_id == unified_id:
            return uc.unified_name
    raise ValueError(f"Unknown unified_id {unified_id}")


class DatasetEmotionMapping:
    def __init__(self, 
        name: str, 
        orig2uni: Dict[Union[str, int], int], 
        orig2name: Dict[Union[str, int], str]
    ):
        self.name = name
        self.orig2uni = orig2uni
        self.orig2name = orig2name

        # Build reverse mappings
        self.uni2orig_ids: Dict[int, List[Union[str, int]]] = {}
        self.uni2orig_names: Dict[int, List[str]] = {}

        for oid, uid in orig2uni.items():
            self.uni2orig_ids.setdefault(uid, []).append(oid)
            oname = orig2name.get(oid)
            if oname:
                self.uni2orig_names.setdefault(uid, []).append(oname)

    def to_unified_id(self, orig_id: Union[str, int]) -> Optional[int]:
        return self.orig2uni.get(orig_id)

    def to_orig_name(self, orig_id: Union[str, int]) -> Optional[str]:
        return self.orig2name.get(orig_id)

    def to_unified_name(self, orig_id: Union[str, int]) -> Optional[str]:
        uid = self.orig2uni.get(orig_id)
        if uid is None:
            return None
        return get_unified_emotion_class_name(uid)

    def unified_name(self, uni_id: int) -> str:
        return get_unified_emotion_class_name(uni_id)

    def to_orig_id(self, unified_id: int) -> Optional[Union[str, int]]:
        if unified_id in self.uni2orig_ids and self.uni2orig_ids[unified_id]:
            return self.uni2orig_ids[unified_id][0]
        return None

###########
# RAVDESS #
###########

RAVDESS_EMOTION_ORIG2NAME = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

RAVDESS_EMOTION_ORIG2UNI = {
    1: 0,    # neutral   → neutral
    2: None, # calm      → ignore
    3: 1,    # happy     → happy
    4: 3,    # sad       → sad
    5: 4,    # angry     → anger
    6: 5,    # fearful   → fear
    7: 6,    # disgust   → disgust
    8: 2,    # surprised → surprise
}

RAVDESS_EMOTION_MAPPING = DatasetEmotionMapping(
    name="ravdess",
    orig2uni=RAVDESS_EMOTION_ORIG2UNI,
    orig2name=RAVDESS_EMOTION_ORIG2NAME
)

########
# MELD #
########

MELD_EMOTION_ORIG2NAME = {
    0: "neutral",
    1: "happy", # Sometimes "joy" in literature, but "happy" matches the unified name
    2: "sadness",
    3: "anger",
    4: "surprise",
    5: "fear",
    6: "disgust",
}

MELD_SENTIMENT_NAME2ORIG = {
    'neutral':  0,
    'positive': 1,
    'negative': 2,
}

MELD_EMOTION_NAME2ORIG = {
    "neutral":  0,
    "joy":      1,
    "sadness":  2,
    "anger":    3,
    "surprise": 4,
    "fear":     5,
    "disgust":  6,
}

MELD_EMOTION_ORIG2UNI = {
    0: 0,  # neutral  → neutral
    1: 1,  # happy    → happy
    2: 3,  # sad      → sad
    3: 4,  # anger    → anger
    4: 2,  # surprise → surprise
    5: 5,  # fear     → fear
    6: 6,  # disgust  → disgust
}

MELD_EMOTION_MAPPING = DatasetEmotionMapping(
    name="meld",
    orig2uni=MELD_EMOTION_ORIG2UNI,
    orig2name=MELD_EMOTION_ORIG2NAME,
)

########
# MEAD #
########

MEAD_EMOTION_ORIG2NAME = {
    0: "neutral",
    1: "angry",
    2: "contempt",
    3: "disgusted",
    4: "happy",
    5: "fear",
    6: "sad",
    7: "surprised",
}

MEAD_EMOTION_NAME2ORIG = {
    v: k
    for k, v 
    in MEAD_EMOTION_ORIG2NAME.items()
}

MEAD_EMOTION_ORIG2UNI = {
    0: 0,  # neutral   → neutral
    1: 4,  # angry     → anger
    2: 7,  # contempt  → contempt
    3: 6,  # disgusted → disgust
    4: 1,  # happy     → happy
    5: 5,  # fear      → fear
    6: 3,  # sad       → sad
    7: 2,  # surprised → surprise
}

MEAD_EMOTION_MAPPING = DatasetEmotionMapping(
    name="mead",
    orig2uni=MEAD_EMOTION_ORIG2UNI,
    orig2name=MEAD_EMOTION_ORIG2NAME,
)

MEAD_CAMERA_ORIG2ID = {
    'down':     0,
    'front':    1,
    'top':      2,
    'left_30':  3,
    'left_60':  4,
    'right_30': 5,
    'right_60': 6,
}

MEAD_CAMERA_ID2ORIG = {
    idx: camera 
    for idx, camera 
    in enumerate(MEAD_CAMERA_ORIG2ID)
}

###########
# CREMA-D #
###########

CREMAD_EMOTION_ORIG2NAME = {
    0: "NEU", # neutral
    1: "HAP", # happy
    2: "SAD", # sad
    3: "ANG", # angry
    4: "FEA", # fear
    5: "DIS", # disgust
}

CREMAD_EMOTION_NAME2ORIG = {
    v: k
    for k, v 
    in CREMAD_EMOTION_ORIG2NAME.items()
}

CREMAD_EMOTION_ORIG2UNI = {
    0: 0,  # neutral → neutral
    1: 1,  # happy   → happy
    2: 3,  # sad     → sad
    3: 4,  # angry   → anger
    4: 5,  # fear    → fear
    5: 6,  # disgust → disgust
}

CREMAD_EMOTION_MAPPING = DatasetEmotionMapping(
    name="cremad",
    orig2uni=CREMAD_EMOTION_ORIG2UNI,
    orig2name=CREMAD_EMOTION_ORIG2NAME,
)

#############
# CelebV-HQ #
#############

CELEBVHQ_EMOTION_ORIG2NAME = {
    0: "neutral",
    1: "anger",
    2: "contempt",
    3: "disgust",
    4: "happy",
    5: "fear",
    6: "sadness",
    7: "surprise",
}

CELEBVHQ_EMOTION_NAME2ORIG = {
    v: k
    for k, v 
    in CELEBVHQ_EMOTION_ORIG2NAME.items()
}

CELEBVHQ_EMOTION_ORIG2UNI = {
    0: 0,  # neutral  → neutral
    1: 4,  # anger    → anger
    2: 7,  # contempt → contempt
    3: 6,  # disgust  → disgust
    4: 1,  # happy    → happy
    5: 5,  # fear     → fear
    6: 3,  # sadness  → sad
    7: 2,  # surprise → surprise
}

CELEBVHQ_EMOTION_MAPPING = DatasetEmotionMapping(
    name="celebvhq",
    orig2uni=CELEBVHQ_EMOTION_ORIG2UNI,
    orig2name=CELEBVHQ_EMOTION_ORIG2NAME,
)

CELEBVHQ_APPEARANCE_NAMES = np.array(['blurry', 'male', 'young', 'chubby', 'pale_skin', 'rosy_cheeks', 'oval_face', 'receding_hairline', 'bald', 'bangs', 'black_hair', 'blonde_hair', 'gray_hair', 'brown_hair', 'straight_hair', 'wavy_hair', 'long_hair', 'arched_eyebrows', 'bushy_eyebrows', 'bags_under_eyes', 'eyeglasses', 'sunglasses', 'narrow_eyes', 'big_nose', 'pointy_nose', 'high_cheekbones', 'big_lips', 'double_chin', 'no_beard', '5_o_clock_shadow', 'goatee', 'mustache', 'sideburns', 'heavy_makeup', 'wearing_earrings', 'wearing_hat', 'wearing_lipstick', 'wearing_necklace', 'wearing_necktie', 'wearing_mask'])
CELEBVHQ_APPEARANCE_NAME2ID = {name: idx for idx, name in enumerate(CELEBVHQ_APPEARANCE_NAMES)}
CELEBVHQ_APPEARANCE_ID2NAME = {idx: name for idx, name in enumerate(CELEBVHQ_APPEARANCE_NAMES)}

CELEBVHQ_ACTION_NAMES = np.array(['blow', 'chew', 'close_eyes', 'cough', 'cry', 'drink', 'eat', 'frown', 'gaze', 'glare', 'head_wagging', 'kiss', 'laugh', 'listen_to_music', 'look_around', 'make_a_face', 'nod', 'play_instrument', 'read', 'shake_head', 'shout', 'sigh', 'sing', 'sleep', 'smile', 'smoke', 'sneer', 'sneeze', 'sniff', 'talk', 'turn', 'weep', 'whisper', 'wink', 'yawn'])
CELEBVHQ_ACTION_NAME2ID = {name: idx for idx, name in enumerate(CELEBVHQ_ACTION_NAMES)}
CELEBVHQ_ACTION_ID2NAME = {idx: name for idx, name in enumerate(CELEBVHQ_ACTION_NAMES)}

########
# CAER #
########

CAER_EMOTION_ORIG2NAME = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

CAER_EMOTION_ORIG2UNI = {
    0: 4,  # anger    → anger
    1: 6,  # disgust  → disgust
    2: 5,  # fear     → fear
    3: 1,  # happy    → happy
    4: 0,  # neutral  → neutral
    5: 3,  # sad      → sad
    6: 2,  # surprise → surprise
}

CAER_EMOTION_MAPPING = DatasetEmotionMapping(
    name="caer",
    orig2uni=CAER_EMOTION_ORIG2UNI,
    orig2name=CAER_EMOTION_ORIG2NAME,
)

############
# AffWild2 #
############

AFFWILD2_EMOTION_ORIG2NAME = {
    0:  "neutral",
    1:  "anger",
    2:  "disgust",
    3:  "fear",
    4:  "happiness",
    5:  "sadness",
    6:  "surprise",
    7:  "other",  # This will be ignored
    -1: "invalid",  # This will be ignored
}

AFFWILD2_EMOTION_ORIG2UNI = {
    0:  0,    # neutral   → neutral
    1:  4,    # anger     → anger
    2:  6,    # disgust   → disgust
    3:  5,    # fear      → fear
    4:  1,    # happiness → happy
    5:  3,    # sadness   → sad
    6:  2,    # surprise  → surprise
    7: -1,    # other     → ignore
   -1: -1     # invalid   → ignore
}

AFFWILD2_EMOTION_MAPPING = DatasetEmotionMapping(
    name="affwild2_expr",
    orig2uni=AFFWILD2_EMOTION_ORIG2UNI,
    orig2name=AFFWILD2_EMOTION_ORIG2NAME,
)

#####################
# EMOTION_INTENSITY #
#####################

@dataclass(frozen=True)
class UnifiedIntensityClass:
    unified_id: int
    unified_name: str

UNIFIED_INTENSITY_CLASSES: List[UnifiedIntensityClass] = [
    UnifiedIntensityClass(0, "weak"),
    UnifiedIntensityClass(1, "normal"),
    UnifiedIntensityClass(2, "strong"),
]

def get_unified_emotion_intensity_name(unified_id: int) -> str:
    for ui in UNIFIED_INTENSITY_CLASSES:
        if ui.unified_id == unified_id:
            return ui.unified_name
    return None


class DatasetIntensityMapping:

    def __init__(self,
        name: str,
        orig2uni: Dict[Union[int, str], Optional[int]],
        orig2name: Dict[Union[int, str], str],
    ):
        self.name = name
        self.orig2uni = orig2uni
        self.orig2name = orig2name

        # Build reverse mappings (unified id -> list of original ids & names)
        self.uni2orig_ids: Dict[int, List[Union[int, str]]] = {}
        self.uni2orig_names: Dict[int, List[str]] = {}

        for oid, uid in orig2uni.items():
            if uid is not None and uid >= 0:
                self.uni2orig_ids.setdefault(uid, []).append(oid)
                oname = orig2name.get(oid)
                if oname:
                    self.uni2orig_names.setdefault(uid, []).append(oname)

    def to_unified_id(self, orig_id: Union[int, str]) -> Optional[int]:
        return self.orig2uni.get(orig_id)

    def to_orig_name(self, orig_id: Union[int, str]) -> Optional[str]:
        return self.orig2name.get(orig_id)

    def to_unified_name(self, orig_id: Union[int, str]) -> Optional[str]:
        uid = self.to_unified_id(orig_id)
        if uid is None:
            return None
        return get_unified_emotion_intensity_name(uid)

    def unified_name(self, uni_id: int) -> str:
        return get_unified_emotion_intensity_name(uni_id)

    def to_orig_id(self, unified_id: int) -> Optional[Union[str, int]]:
        if unified_id in self.uni2orig_ids and self.uni2orig_ids[unified_id]:
            return self.uni2orig_ids[unified_id][0]
        return None

###########
# RAVDESS #
###########

RAVDESS_INTENSITY_ORIG2NAME = {
    # weak (0) not in RAVDESS, so omitted
    1: "normal",
    2: "strong",
}

RAVDESS_INTENSITY_ORIG2UNI = {
    # weak (0) missing
    1: 1,  # normal -> normal (1)
    2: 2,  # strong -> strong (2)
}

RAVDESS_INTENSITY_MAPPING = DatasetIntensityMapping(
    name="ravdess",
    orig2uni=RAVDESS_INTENSITY_ORIG2UNI,
    orig2name=RAVDESS_INTENSITY_ORIG2NAME,
)

###########
# CREMA-D #
###########

CREMAD_INTENSITY_ORIG2NAME = {
    0: 'LO', # "weak"
    1: 'MD', # "normal",
    2: 'HI', # "strong",
    3: 'XX', # "unspecified",
}

CREMAD_INTENSITY_NAME2ORIG = {
    v: k
    for k, v 
    in CREMAD_INTENSITY_ORIG2NAME.items()
}

CREMAD_INTENSITY_ORIG2UNI = {
    0: 0,     # LO -> weak (0)
    1: 1,     # MD -> normal (1)
    2: 2,     # HI -> strong (2)
    3: None,  # XX -> ignore
}

CREMAD_INTENSITY_MAPPING = DatasetIntensityMapping(
    name="cremad",
    orig2uni=CREMAD_INTENSITY_ORIG2UNI,
    orig2name=CREMAD_INTENSITY_ORIG2NAME,
)

########
# MEAD #
########

MEAD_INTENSITY_ORIG2NAME = {
    1: "weak",
    2: "normal",
    3: "strong",
}

MEAD_INTENSITY_ORIG2UNI = {
    1: 0,  # weak -> weak (0)
    2: 1,  # normal -> normal (1)
    3: 2,  # strong -> strong (2)
}

MEAD_INTENSITY_MAPPING = DatasetIntensityMapping(
    name="mead",
    orig2uni=MEAD_INTENSITY_ORIG2UNI,
    orig2name=MEAD_INTENSITY_ORIG2NAME,
)