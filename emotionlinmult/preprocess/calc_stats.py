from pathlib import Path
import json
from tqdm import tqdm


def count_samples():
    datasets = {
        'AFEWVA': 'data/db_processed/AFEW-VA/count_samples_face.json', 
        'AffWild2_expr': 'data/db_processed/Aff-Wild2/count_samples_expr.json', 
        'AffWild2_va': 'data/db_processed/Aff-Wild2/count_samples_va.json', 
        #'CelebVHQ': 'data/db_processed/CelebV-HQ/count_samples.json', 
        'CREMAD_expr': 'data/db_processed/CREMA-D/count_samples_expr.json', 
        'CREMAD_int': 'data/db_processed/CREMA-D/count_samples_int.json', 
        'MEAD': 'data/db_processed/MEAD/count_samples.json', 
        #'MELD': 'data/db_processed/MELD/count_samples.json', 
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


def calculate_weights(class_counts: dict[int, int]):
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    return {i: (total / (num_classes * class_counts[i])) for i in class_counts}


def distribution(output_dir: str = 'data/db_processed'):
    from emotionlinmult.train.datamodule import MultiDatasetModule

    config = {
        'feature_list': ['wavlm_baseplus', 'clip'], 
        'target_list': ['emotion_class', 'emotion_intensity'], 
        'datasets': {
            'train': ['ravdess', 'mosei', 'cremad_expr', 'cremad_int', 'affwild2_expr', 'affwild2_va', 'afewva'],
            'valid': ['ravdess', 'mosei', 'cremad_expr', 'cremad_int', 'affwild2_expr', 'affwild2_va', 'afewva'],
            'test': ['ravdess', 'mosei', 'cremad_expr', 'cremad_int', 'mead', 'affwild2_expr', 'affwild2_va', 'afewva']
        },
        'proportion_sampling': True,
        'num_workers': 0,
        'batch_size': 32,
    }

    datamodule = MultiDatasetModule(config)
    datamodule.setup()

    # sum observed emotion classes during one epoch:
    emotion_class_counts = {i: 0 for i in range(7)}
    emotion_intensity_counts = {i: 0 for i in range(3)}

    train_dataloader = datamodule.train_dataloader()
    for batch in tqdm(train_dataloader, total=len(train_dataloader)):

        for emotion_class in batch['emotion_class'][batch['emotion_class_mask']]:
            emotion_class_counts[int(emotion_class)] += 1
        for emotion_intensity in batch['emotion_intensity'][batch['emotion_intensity_mask']]:
            emotion_intensity_counts[int(emotion_intensity)] += 1

        print(
            f" ravdess: {sum([elem == 'RAVDESS' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
            f" mosei: {sum([elem == 'MOSEI' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
            f" cremad_expr: {sum([elem == 'CREMA-D_expr' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
            f" cremad_int: {sum([elem == 'CREMA-D_int' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
            #f" mead: {sum([elem == 'MEAD' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
            f" affwild2_expr: {sum([elem == 'AffWild2_expr' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
            f" affwild2_va: {sum([elem == 'AffWild2_va' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
            f" afewva: {sum([elem == 'AFEW-VA_face' for elem in batch['dataset']]) / len(batch['dataset'])}"
        )

    name = '-'.join(config['datasets']['train'])
    with open(f'{output_dir}/observed_emotion_class_counts_{name}.json', 'w') as f:
        json.dump(emotion_class_counts, f)
    with open(f'{output_dir}/observed_emotion_intensity_counts_{name}.json', 'w') as f:
        json.dump(emotion_intensity_counts, f)

    emotion_class_weights = calculate_weights(emotion_class_counts)
    emotion_intensity_weights = calculate_weights(emotion_intensity_counts)

    with open(f'{output_dir}/observed_emotion_class_weights_{name}.json', 'w') as f:
        json.dump(emotion_class_weights, f)
    with open(f'{output_dir}/observed_emotion_intensity_weights_{name}.json', 'w') as f:
        json.dump(emotion_intensity_weights, f)

if __name__ == '__main__':
    distribution()