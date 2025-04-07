import pickle
from pathlib import Path

max_size = 0
sizes = []
for path in sorted(list(Path('data/db_processed/AFEW-VA/fabnet').glob('*.pkl'))):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        frame_ids = data[0]
        features = data[1]

    max_size = max(max_size, features.shape[0])
    sizes.append(features.shape[0])

print('Max number of frames:', max_size)
print(min(sizes))
print(sizes)