from pathlib import Path


DB = Path('data/db/MELD')
DB_PROCESSED = Path('data/db_processed/MELD')

ANNOTATION_PATHS = {
    'train': DB / 'train_sent_emo.csv',
    'valid': DB / 'dev_sent_emo.csv',
    'test': DB / 'test_sent_emo.csv'
}

SUBSET_NAME = {
    'train_splits': 'train',
    'dev_splits_complete': 'valid',
    'output_repeated_splits_test': 'test'
}

