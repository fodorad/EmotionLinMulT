import os
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from exordium.video.fabnet import FabNetWrapper
from exordium.video.opengraphau import OpenGraphAuWrapper
from exordium.video.clip import ClipWrapper


DB = Path('data/db/Aff-Wild2')
DB_PROCESSED = Path('data/db_processed/Aff-Wild2')


def test_video(video_dir):
    from decord import VideoReader
    sample_id = Path(video_dir).stem
    video_path = next((path for path in video_paths if path.stem == sample_id.replace('_left', '').replace('_right', '')), None)
    try:
        vr = VideoReader(video_path)
        len(vr)
        return video_path
    except:
        new_path = DB_PROCESSED / 'video' / f"{sample_id}.mp4"
        if not new_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            os.system(f"ffmpeg -i {str(video_path)} -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k {str(new_path)}")
    return new_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess Aff-Wild2 visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=600, help='end index')
    args = parser.parse_args()

    cropped_aligned_dirs = sorted(
        list((DB / 'cropped_aligned' / 'cropped_aligned').glob("*")) + \
        list((DB / 'cropped_aligned' / 'cropped_aligned_new_50_vids').glob("*"))
    )[args.start:args.end][::-1]
    video_paths = sorted(
        list((DB / 'videos').glob('**/*.mp4')) + \
        list((DB / 'videos').glob('**/*.avi'))
    )
    print(f"Found {len(cropped_aligned_dirs)} face dirs ({args.start}-{args.end})...")
    print(f"Found {len(video_paths)} videos ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}") 
    fabnet_extractor = FabNetWrapper(gpu_id=args.gpu_id)
    au_extractor = OpenGraphAuWrapper(gpu_id=args.gpu_id)
    clip_extractor = ClipWrapper(gpu_id=args.gpu_id)

    for i, video_dir in tqdm(enumerate(cropped_aligned_dirs), total=len(cropped_aligned_dirs), desc='Face videos'):
        start_time = time.time()
        sample_id = Path(video_dir).stem
        
        if sample_id == '.DS_Store': continue

        if (DB_PROCESSED / 'fabnet' / f'{sample_id}.pkl').exists() and \
           (DB_PROCESSED / 'opengraphau' / f'{sample_id}.pkl').exists() and \
           (DB_PROCESSED / 'clip' / f'{sample_id}.npy').exists():
            continue

        print(f'Sample id: {sample_id}')

        try:
            face_paths = sorted(list(video_dir.glob('*.jpg')))

            if not (DB_PROCESSED / 'clip' / f'{sample_id}.npy').exists():
                video_path = test_video(video_dir)
                features = clip_extractor.extract_from_video(video_path, verbose=True, output_path=DB_PROCESSED / 'clip' / f'{sample_id}.npy')
                print('clip features:', features.shape)
            
            if not (DB_PROCESSED / 'fabnet' / f'{sample_id}.pkl').exists():
                ids, embeddings = fabnet_extractor.dir_to_feature(face_paths, batch_size=15, verbose=True, output_path=DB_PROCESSED / 'fabnet' / f'{sample_id}.pkl')
                print('fabnet embeddings:', embeddings.shape)

            if not (DB_PROCESSED / 'opengraphau' / f'{sample_id}.pkl').exists():
                ids, au = au_extractor.dir_to_feature(face_paths, batch_size=15, verbose=True, output_path=DB_PROCESSED / 'opengraphau' / f'{sample_id}.pkl')
                print('au:', au.shape)

        except Exception as e:
            with open(f"data/db_processed/Aff-Wild2/skip_feat.txt", "a") as f:
                f.write(f"{video_dir}\t{video_path}\t{sample_id}\t{str(e)}\n")
            print(e)
            pass

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:03f} seconds")