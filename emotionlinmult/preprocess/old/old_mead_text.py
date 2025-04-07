import numpy as np
from exordium.text.xml_roberta import XmlRobertaWrapper
from tqdm import tqdm
from pathlib import Path

DB_PROCESSED = Path("data/db_processed/MEAD")

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

file_path = 'data/db_processed/transcript.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

text_model = XmlRobertaWrapper()

for line in tqdm(lines, total=len(lines), desc="Samples"):
    parts = line.split(sep=" ")
    sample_id_parts = parts[0].split("_")
    participant_id = sample_id_parts[0]
    label_intensity = sample_id_parts[2]
    label_emotion = EMOTION_TO_ID[sample_id_parts[3]]
    video_id = sample_id_parts[4]
    sample_id = f'{label_emotion}-{label_intensity}-{video_id}'
    transcript = " ".join(parts[1:])
    output_file = DB_PROCESSED / participant_id / "xml_roberta" / sample_id
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if (output_file.parent / f'{output_file.name}.npy').exists(): continue
    embedding = text_model(transcript)
    vector = np.squeeze(embedding) # (768,)
    np.save(str(output_file), vector)
    print("xml_roberta:", str(output_file))