from pathlib import Path
from prettytable import PrettyTable
import numpy as np
from emotionlinmult.preprocess.MELD import DB_PROCESSED
from emotionlinmult.preprocess import CLIP_TIME_DIM
from tqdm import tqdm


clip_paths = sorted(list([elem for elem in (DB_PROCESSED / 'clip').glob('**/*.npy')]))
print(f"Total clips: {len(clip_paths)}")

table = PrettyTable()
table.field_names = ["Length (mean ± std)", 'Min', 'Max', 'Longer than 10 sec']

lengths = [
    np.load(clip_path).shape[0] 
    for clip_path in tqdm(clip_paths, total=len(clip_paths), desc='MELD clips')
]

mean_length = np.mean(lengths)
std_length = np.std(lengths)
min_length = np.min(lengths)
max_length = np.max(lengths)
longer = np.sum(np.array(lengths) > CLIP_TIME_DIM)
table.add_row([f"{mean_length:.2f} ± {std_length:.2f}", min_length, max_length, longer])

print(table)