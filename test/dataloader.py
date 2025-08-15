from emotionlinmult.train.datamodule import MultiDatasetModule


config = {
    'feature_list': ['wavlm_baseplus', 'clip'], 
    'target_list': ['emotion_class', 'emotion_intensity'], 
    'datasets': {
        'train': ['ravdess', 'mosei', 'cremad_expr'], #, 'celebv_hq', 'cremad', 'meld', 'mosei', 'mead'],
        'valid': ['ravdess', 'mosei'],
        'test': ['ravdess']
    },
    'proportion_sampling': True,
    'num_workers': 2,
}

datamodule = MultiDatasetModule(config)
datamodule.setup()

train_dataset = datamodule.train_dataset
print('size:', len(train_dataset))

for i, sample in enumerate(train_dataset):
    print(f"[Sample {i}]")
    if i == 2: break

train_dataloader = datamodule.train_dataloader()
#print('size:', len(train_dataloader))

for i, batch in enumerate(train_dataloader):
    print(f"[Batch {i}]" +
    f" ravdess: {sum([elem == 'RAVDESS' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
    f" mosei: {sum([elem == 'MOSEI' for elem in batch['dataset']]) / len(batch['dataset'])} | " +
    f" cremad_expr: {sum([elem == 'CREMA-D_expr' for elem in batch['dataset']]) / len(batch['dataset'])}")
    pass
    if i == 2: break
    