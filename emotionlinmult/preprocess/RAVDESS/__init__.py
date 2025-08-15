from pathlib import Path

DB = Path('data/db/RAVDESS')
DB_PROCESSED = Path("data/db_processed/RAVDESS")

PARTICIPANT_IDS = {
    'train': [f"{i:02d}" for i in range(1, 25)][0:16],
    'valid': [f"{i:02d}" for i in range(1, 25)][16:20],
    'test':  [f"{i:02d}" for i in range(1, 25)][20:],
}
