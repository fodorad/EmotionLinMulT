from pathlib import Path
import json



def count_samples():
    datasets = {
        'AFEWVA': 'data/db_processed/AFEW-VA/count_samples_face.json', 
        'AffWild2_expr': 'data/db_processed/Aff-Wild2/count_samples_expr.json', 
        'AffWild2_va': 'data/db_processed/Aff-Wild2/count_samples_va.json', 
        'CelebVHQ': 'data/db_processed/CelebV-HQ/count_samples.json', 
        'CREMAD_expr': 'data/db_processed/CREMA-D/count_samples_expr.json', 
        'CREMAD_int': 'data/db_processed/CREMA-D/count_samples_int.json', 
        'MEAD': 'data/db_processed/MEAD/count_samples.json', 
        'MELD': 'data/db_processed/MELD/count_samples.json', 
        'MOSEI': 'data/db_processed/MOSEI/count_samples.json', 
        'RAVDESS': 'data/db_processed/RAVDESS/count_samples.json', 
    }

    total_size = 0
    for dataset in datasets:
        with open(datasets[dataset], 'r') as f:
            data = json.load(f)
        print(f"{dataset}: {data['train']} | {data['valid']} | {data['test']}")

        if dataset == 'MEAD':
            total_size += 10000
        else:
            total_size += data['train']
        
    print(f"Total size: {total_size}")
    

def distribution():
    from emotionlinmult.train.datamodule import MultiDatasetModule

    config = {
        'feature_list': ['wavlm_baseplus', 'clip'], 
        'target_list': ['emotion_class', 'emotion_intensity'], 
        'datasets': {
            'train': ['ravdess', 'mosei', 'cremad_expr', 'cremad_int', 'meld', 'mead', 'celebvhq', 'affwild2_expr', 'affwild2_va', 'afewva'],
            'valid': ['ravdess', 'mosei', 'cremad_expr', 'cremad_int', 'meld', 'mead', 'celebvhq', 'affwild2_expr', 'affwild2_va', 'afewva'],
            'test': ['ravdess', 'mosei', 'cremad_expr', 'cremad_int', 'meld', 'mead', 'celebvhq', 'affwild2_expr', 'affwild2_va', 'afewva']
        },
        'proportion_sampling': True,
    }

    datamodule = MultiDatasetModule(config)
    datamodule.setup()

    # sum observed emotion classes during one epoch:
    emotion_class_counts = {i: 0 for i in range(8)}
    emotion_intensity_counts = {i: 0 for i in range(3)}

    train_dataloader = datamodule.train_dataloader()
    for i, batch in enumerate(train_dataloader):

        for emotion_class in batch['emotion_class'][batch['emotion_class_mask']]:
            emotion_class_counts[int(emotion_class)] += 1
        for emotion_intensity in batch['emotion_intensity'][batch['emotion_intensity_mask']]:
            emotion_intensity_counts[int(emotion_intensity)] += 1

        print(f"[Batch {i}]" +
        f" ravdess: {sum([elem == 'RAVDESS' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" mosei: {sum([elem == 'MOSEI' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" cremad_expr: {sum([elem == 'CREMA-D_expr' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" cremad_int: {sum([elem == 'CREMA-D_int' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" meld: {sum([elem == 'MELD' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" mead: {sum([elem == 'MEAD' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" celebvhq: {sum([elem == 'CelebV-HQ' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" affwild2_expr: {sum([elem == 'AffWild2_expr' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" affwild2_va: {sum([elem == 'AffWild2_va' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
        f" afewva: {sum([elem == 'AFEW-VA_face' for elem in batch['dataset']]) / len(batch['dataset'])}")

    with open('observed_emotion_class_counts.json', 'w') as f:
        json.dump(emotion_class_counts, f)
    with open('observed_emotion_intensity_counts.json', 'w') as f:
        json.dump(emotion_intensity_counts, f)


if __name__ == '__main__':
    distribution()