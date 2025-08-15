from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pickle
from exordium.text.xml_roberta import XmlRobertaWrapper
from emotionlinmult.preprocess.MOSEI import DB_PROCESSED


def get_annotation(csv_path: str, subset: str):
    assert subset in ['train', 'valid', 'test'], f"Invalid subset: {subset}"

    df = pd.read_csv(csv_path, sep=',')

    annotation = {}
    for _, row in df.iterrows():
        sample_id = f"{row['video_id']}_{row['clip_id']}"
        entry = {
            'sentiment': float(row['label']),
            'text': row['text'].strip()
        }

        if row['mode'] == subset:
            annotation[sample_id] = entry

    return annotation


if __name__ == "__main__":

    xml_roberta = XmlRobertaWrapper()

    output_path = DB_PROCESSED / 'xml_roberta' / f'mosei_xml_roberta.pkl'
    samples = {}

    for subset in ['train', 'valid', 'test']:

        annotation = get_annotation(DB_PROCESSED / 'mosei_label.csv', subset)
        subset_ids = sorted(list(annotation.keys()))

        for sample_id in tqdm(subset_ids, total=len(subset_ids), desc=f'{subset}'):

            sentiment = annotation[sample_id]['sentiment'] # float
            text = annotation[sample_id]['text'] # str
            token_embeddings, _ = xml_roberta(text)
            token_embeddings = token_embeddings.detach().cpu().numpy().squeeze(0) # (1, T, F) -> (T, F)

            assert token_embeddings.ndim == 2 and token_embeddings.shape[0] != 0 and token_embeddings.shape[1] == 768

            samples[sample_id] = {
                'xml_roberta': token_embeddings, # (T, F) or None
                'sentiment': sentiment, # ()
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)