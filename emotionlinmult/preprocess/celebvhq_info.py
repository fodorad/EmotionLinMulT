from pathlib import Path
from tqdm import tqdm
from pprint import pprint
import numpy as np
import pandas as pd
from exordium.video.detection import VideoDetections
from exordium.video.tracker import IouTracker


tracker_path = Path('data/db_processed/CelebV-HQ/tracker')
vdet_paths = sorted(list(tracker_path.glob('*.vdet')))[:2000]

track_data = []
for vdet_path in tqdm(vdet_paths, total=len(vdet_paths)):
    videodetections = VideoDetections().load(vdet_path)
    track = IouTracker(max_lost=30).label(videodetections).merge().get_center_track()
    if track is None: continue
    track_data.append({'path': vdet_path, 'len': len(track)})

df = pd.DataFrame(track_data)
df.to_csv('track_data.csv', index=False)
