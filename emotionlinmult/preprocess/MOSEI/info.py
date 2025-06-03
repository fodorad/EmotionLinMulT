import pickle
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
from tqdm import tqdm

DB_PROCESSED = Path("data/db_processed/MOSEI")

xml_roberta_path = DB_PROCESSED / 'xml_roberta' / 'mosei_xml_roberta.pkl'
with open(xml_roberta_path, 'rb') as f:
    xml_roberta_dict = pickle.load(f)

roberta_tokens = []
num_extreme_long = 0
for key in list(xml_roberta_dict.keys()):
    length = xml_roberta_dict[key]['xml_roberta'].shape[0]
    if length > 120: num_extreme_long += 1
    roberta_tokens.append(length)

table = PrettyTable()
table.field_names = ["num samples", "Min", "Max", "Mean ± Std", "Extreme Long (>120)"]
table.add_row([len(roberta_tokens), np.min(roberta_tokens), np.max(roberta_tokens), f"{np.mean(roberta_tokens):.2f} ± {np.std(roberta_tokens):.2f}", num_extreme_long])
print(table) 

'''
+-------------+-----+-----+---------------+---------------------+
| num samples | Min | Max |   Mean ± Std  | Extreme Long (>120) |
+-------------+-----+-----+---------------+---------------------+
|    22856    |  3  | 442 | 28.07 ± 17.55 |          45         |
+-------------+-----+-----+---------------+---------------------+
'''

clip_paths = sorted(list([elem for elem in (DB_PROCESSED / 'clip').glob('**/*.npy')]))
print(f"Total clips: {len(clip_paths)}")

table = PrettyTable()
table.field_names = ["Length (mean ± std)", 'Min', 'Max']

lengths = [np.load(clip_path).shape[0] for clip_path in clip_paths]
mean_length = np.mean(lengths)
std_length = np.std(lengths)
min_length = np.min(lengths)
max_length = np.max(lengths)
table.add_row([f"{mean_length:.2f} ± {std_length:.2f}", min_length, max_length])

print(table)
'''
Total clips: 22856

+---------------------+-----+------+
| Length (mean ± std) | Min | Max  |
+---------------------+-----+------+
|   231.23 ± 151.89   |  4  | 3253 |
+---------------------+-----+------+
'''