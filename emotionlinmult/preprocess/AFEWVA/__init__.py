from pathlib import Path
import numpy as np


DB = Path('data/db/AFEW-VA')
DB_PROCESSED = Path('data/db_processed/AFEW-VA')

SUBSET_IDS = {
    'train': np.arange(1, 421),   # 70%
    'valid': np.arange(421, 511), # 15%
    'test': np.arange(511, 601),  # 15%
}