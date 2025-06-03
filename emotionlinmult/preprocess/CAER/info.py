import pickle
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
from tqdm import tqdm

DB = Path("data/db/CAER")
DB_PROCESSED = Path("data/db_processed/CAER")

xml_roberta_path = DB_PROCESSED / 'xml_roberta' / 'caer_xml_roberta.pkl'
with open(xml_roberta_path, 'rb') as f:
    xml_roberta_dict = pickle.load(f)

roberta_tokens = []
num_extreme_long = 0
for key in list(xml_roberta_dict.keys()):
    length = xml_roberta_dict[key]['xml_roberta'].shape[0]
    if length > 120: num_extreme_long += 1
    roberta_tokens.append(length)

table = PrettyTable()
table.field_names = ["Min", "Max", "Mean ± Std", "Extreme Long (>120)"]
table.add_row([np.min(roberta_tokens), np.max(roberta_tokens), f"{np.mean(roberta_tokens):.2f} ± {np.std(roberta_tokens):.2f}", num_extreme_long])
print(table) 

'''
+-----+-----+---------------+---------------------+
| Min | Max |   Mean ± Std  | Extreme Long (>120) |
+-----+-----+---------------+---------------------+
|  2  | 512 | 14.35 ± 18.74 |          22         |
+-----+-----+---------------+---------------------+
'''

clip_paths = sorted(list([elem for elem in (DB_PROCESSED / 'clip').glob('**/*.npy')]))
print(f"Total clips: {len(clip_paths)}")

table = PrettyTable()
table.field_names = ["Subset", "Label", "Length (mean ± std)", 'Min', 'Max']
for subset in ['train', 'validation', 'test']:
    subset_clip_paths = [clip_path for clip_path in clip_paths if clip_path.parent.parent.name == subset]
    subset_labels = [clip_path.parent.name for clip_path in subset_clip_paths]
    unique_labels = set(subset_labels)
    
    for label in unique_labels:
        label_clip_paths = [clip_path for clip_path in subset_clip_paths if clip_path.parent.name == label]
        lengths = [np.load(clip_path).shape[0] for clip_path in label_clip_paths]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        table.add_row([subset, label, f"{mean_length:.2f} ± {std_length:.2f}", min_length, max_length])

print(table)

'''
Total clips: 13175
+------------+----------+---------------------+-----+-----+
|   Subset   |  Label   | Length (mean ± std) | Min | Max |
+------------+----------+---------------------+-----+-----+
|   train    |   Sad    |    94.38 ± 61.59    |  35 | 507 |
|   train    | Disgust  |    84.23 ± 56.88    |  35 | 674 |
|   train    | Neutral  |    83.74 ± 49.76    |  35 | 596 |
|   train    |  Happy   |    82.05 ± 51.18    |  35 | 753 |
|   train    | Surprise |    81.58 ± 50.32    |  35 | 519 |
|   train    |   Fear   |    94.04 ± 66.40    |  35 | 609 |
|   train    |  Anger   |    85.56 ± 51.49    |  35 | 472 |
| validation |   Sad    |    102.51 ± 76.27   |  35 | 475 |
| validation | Disgust  |    98.58 ± 78.34    |  38 | 593 |
| validation | Neutral  |    81.59 ± 44.76    |  35 | 469 |
| validation |  Happy   |    83.67 ± 56.44    |  35 | 438 |
| validation | Surprise |    86.80 ± 56.32    |  37 | 449 |
| validation |   Fear   |    90.47 ± 54.21    |  36 | 321 |
| validation |  Anger   |    92.44 ± 50.89    |  35 | 274 |
|    test    |   Sad    |    95.68 ± 58.62    |  36 | 342 |
|    test    | Disgust  |    82.57 ± 56.33    |  35 | 492 |
|    test    | Neutral  |    82.35 ± 50.03    |  35 | 474 |
|    test    |  Happy   |    81.39 ± 51.24    |  35 | 427 |
|    test    | Surprise |    80.60 ± 50.45    |  35 | 329 |
|    test    |   Fear   |    96.95 ± 58.08    |  35 | 281 |
|    test    |  Anger   |    87.82 ± 51.61    |  36 | 413 |
+------------+----------+---------------------+-----+-----+
'''