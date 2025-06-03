from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pickle
from exordium.text.xml_roberta import XmlRobertaWrapper


DB = Path("data/db/MELD")
DB_PROCESSED = Path("data/db_processed/MELD")

ANNOTATION_PATHS = {
    'train': DB / 'train_sent_emo.csv',
    'valid': DB / 'dev_sent_emo.csv',
    'test': DB / 'test_sent_emo.csv'
}

SENTIMENT_TO_CLASS = {
    'neutral': 0,
    'positive': 1,
    'negative': 2,
}

EMOTION_TO_CLASS = {
    'neutral': 0,
    'joy': 1,
    'sadness': 2,
    'anger': 3,
    'surprise': 4,
    'fear': 5,
    'disgust': 6,
}


def get_annotation(csv_path: str):
    df = pd.read_csv(csv_path, sep=',')

    annotation = {}
    for _, row in df.iterrows():
        sample_id = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
        entry = {
            'sentiment': SENTIMENT_TO_CLASS[row['Sentiment'].strip().lower()], # sentiment
            'emotion_class': EMOTION_TO_CLASS[row['Emotion'].strip().lower()], # emotion
            'text': row['Utterance'], # utterance
        }

        annotation[sample_id] = entry

    return annotation


if __name__ == "__main__":

    xml_roberta = XmlRobertaWrapper()

    output_path = DB_PROCESSED / 'xml_roberta' / f'meld_xml_roberta.pkl'
    samples = {'train': {}, 'valid': {}, 'test': {}}

    for subset in ['train', 'valid', 'test']:

        annotation = get_annotation(ANNOTATION_PATHS[subset])
        subset_ids = sorted(list(annotation.keys()))

        for sample_id in tqdm(subset_ids, total=len(subset_ids), desc=f'{subset}'):

            sentiment = annotation[sample_id]['sentiment'] # float
            emotion_class = annotation[sample_id]['emotion_class'] # float
            text = annotation[sample_id]['text'] # str
            token_embeddings, _ = xml_roberta(text)
            token_embeddings = token_embeddings.detach().cpu().numpy().squeeze(0) # (1, T, F) -> (T, F)

            assert token_embeddings.ndim == 2 and token_embeddings.shape[0] != 0 and token_embeddings.shape[1] == 768

            samples[subset][sample_id] = {
                'xml_roberta': token_embeddings, # (T, F) or None
                'sentiment': sentiment, # ()
                'emotion_class': emotion_class, # ()
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)