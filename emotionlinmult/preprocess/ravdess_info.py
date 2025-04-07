import pickle
import numpy as np
from pathlib import Path

sizes = []
for path in sorted(list(Path('data/db_processed/RAVDESS/fabnet').glob('*.pkl'))):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        frame_ids = data[0]
        features = data[1]
    sizes.append(features.shape[0])
print('fabnet time_dim:', min(sizes), max(sizes))

sizes = []
for path in sorted(list(Path('data/db_processed/RAVDESS/egemaps_lld').glob('*.npy'))):
    features = np.load(str(path))
    sizes.append(features.shape[0])
print('egemaps lld time_dim:', min(sizes), max(sizes))

sizes = []
for path in sorted(list(Path('data/db_processed/RAVDESS/wavlm_baseplus').glob('*.pkl'))):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    features = data[-1] # last layer
    sizes.append(features.shape[0])
print('wavlm time_dim:', min(sizes), max(sizes))