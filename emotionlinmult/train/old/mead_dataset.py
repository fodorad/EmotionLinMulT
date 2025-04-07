import yaml
import pickle
from pathlib import Path
from functools import partial
import torch
import torch.nn.functional as F
import psutil
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from exordium.utils.normalize import standardization, get_mean_std


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_mean_std(path: str | Path) -> tuple[list[float], list[float]]:
    data = np.load(path)
    return data['mean'], data['std']


def generate_uuids(db_processed: str, participant_ids: list[str]) -> list[str]:
    ignore_list_path = Path(db_processed).parent / 'ignore_uuids.txt'

    with open(ignore_list_path, 'r') as file:
        ignore_list = file.readlines()

    ignore_list = set([line.strip() for line in ignore_list])

    uuids = []
    for participant_id in participant_ids:
        participant_path = Path(db_processed) / participant_id
        feature_path = participant_path / "opengraphau"
        files = list(feature_path.glob("*-*-*-*.pkl"))

        for file in files:

            if f'{file.parents[1].name}-{"-".join(file.stem.split("-")[1:])}' in ignore_list:
                continue

            # participant_id-feature_name-camera_angle-class-intensity-file_id-suffix
            uuid = f'{file.parents[1].name}-{file.parent.name}-{file.stem}-{file.suffix}'
            uuids.append(uuid)

    uuids = sorted(uuids)
    return uuids


def pad_or_crop_np(tensor: np.ndarray, max_length: int) -> np.ndarray:
    N, D = tensor.shape

    if N < max_length:
        # Pad the tensor with zeros if N < max_length
        pad_width = ((0, max_length - N), (0, 0))
        tensor_padded = np.pad(tensor, pad_width, mode='constant', constant_values=0)
        return tensor_padded
    else:
        # Crop the tensor if N > max_length
        tensor_cropped = tensor[:max_length, :]
        return tensor_cropped


def pad_or_crop(tensor: torch.Tensor, max_length: int) -> torch.Tensor:
    N, D = tensor.shape

    if N < max_length:
        # Pad the tensor with zeros if N < max_length
        pad_amount = max_length - N
        # Padding dimensions for torch.nn.functional.pad is (padding_left, padding_right, padding_top, padding_bottom)
        # Here we pad only on the bottom, so (0, 0) for width (left and right), and (0, pad_amount) for height (top and bottom)
        tensor_padded = F.pad(tensor, (0, 0, 0, pad_amount), mode='constant', value=0)
        return tensor_padded
    else:
        # Crop the tensor if N > max_length
        tensor_cropped = tensor[:max_length, :]
        return tensor_cropped



class UUID:

    def __init__(self, uuid_str):
        parts = uuid_str.split('-')
        if len(parts) != 7:
            raise ValueError("UUID string must have exactly 7 parts separated by '-'")

        self.participant_id = parts[0]
        self.feature_name = parts[1]
        self.camera_angle = parts[2]
        self.emotion_class = int(parts[3])
        self.emotion_intensity = int(parts[4])
        self.file_id = parts[5]
        self.suffix = parts[6]

    def __repr__(self):
        return (f"UUID(participant_id={self.participant_id}, feature_name={self.feature_name}, "
                f"camera_angle={self.camera_angle}, emotion_class={self.emotion_class}, "
                f"emotion_intensity={self.emotion_intensity}, file_id={self.file_id}, suffix={self.suffix})")


class FabnetDataset(Dataset):

    def __init__(self, config, uuids):
        self.config = config
        self.participant_ids = config.get('participant_ids')
        self.uuids = uuids
        self.window_size: int = np.rint(config.get('window_size_sec') * 30).astype(int) # sec * fps

    def __len__(self):
        return len(self.uuids)

    def load_file(self, uuid):
        o = UUID(uuid)
        path = Path(self.config.get('db_processed')) / o.participant_id / 'fabnet' / f'{o.camera_angle}-{o.emotion_class}-{o.emotion_intensity}-{o.file_id}.pkl'

        if not path.exists():
            raise FileNotFoundError(str(path))

        with open(path, 'rb') as f:
            data = pickle.load(f)

        X = torch.FloatTensor(data[1])

        Y = {
            'emotion_class': torch.LongTensor([o.emotion_class]),
            'emotion_intensity': torch.LongTensor([o.emotion_intensity-1]),
            'file_id': o.file_id
        }

        return X, Y

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        X, Y = self.load_file(uuid)
        X = pad_or_crop(X, self.window_size)
        return X, Y # (N, 256)


class OpenGraphAuDataset(Dataset):

    def __init__(self, config, uuids):
        self.config = config
        self.participant_ids = config.get('participant_ids')
        self.uuids = uuids
        self.window_size: int = np.rint(config.get('window_size_sec') * 30).astype(int) # sec * fps
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.uuids)

    def get_transform(self):
        path = Path(self.config.get('db_processed')).parent / 'norm' / 'mean_std_opengraphau.npz'
        if not path.exists(): return None
        mean, std = load_mean_std(path)
        return partial(standardization, mean=mean, std=std)

    def load_file(self, uuid):
        o = UUID(uuid)
        path = Path(self.config.get('db_processed')) / o.participant_id / 'opengraphau' / f'{o.camera_angle}-{o.emotion_class}-{o.emotion_intensity}-{o.file_id}.pkl'

        if not path.exists():
            raise FileNotFoundError(str(path))

        with open(path, 'rb') as f:
            data = pickle.load(f)

        X = torch.FloatTensor(data[1])

        Y = {
            'emotion_class': torch.LongTensor([o.emotion_class]),
            'emotion_intensity': torch.LongTensor([o.emotion_intensity-1]),
            'file_id': o.file_id
        }

        return X, Y

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        X, Y = self.load_file(uuid)

        if self.transform:
            X = self.transform(X)

        X = pad_or_crop(X, self.window_size)
        return X, Y # (N, 256)


class EgemapsLldDataset(Dataset):

    def __init__(self, config, uuids):
        self.config = config
        self.participant_ids = config.get('participant_ids')
        self.uuids = uuids
        self.window_size: int = np.rint(config.get('window_size_sec') * 100).astype(int) # sec * sr
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.uuids)

    def get_transform(self):
        path = Path(self.config.get('db_processed')).parent / 'norm' / 'mean_std_egemaps_lld.npz'
        if not path.exists(): return None
        mean, std = load_mean_std(path)
        return partial(standardization, mean=mean, std=std)

    def load_file(self, uuid):
        o = UUID(uuid)
        path = Path(self.config.get('db_processed')) / o.participant_id / 'egemaps' / f'{o.emotion_class}-{o.emotion_intensity}-{o.file_id}.npz'

        if not path.exists():
            raise FileNotFoundError(str(path))

        X = torch.FloatTensor(np.load(path)['lld'])

        Y = {
            'emotion_class': torch.LongTensor([o.emotion_class]),
            'emotion_intensity': torch.LongTensor([o.emotion_intensity-1]),
            'file_id': o.file_id
        }

        return X, Y

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        X, Y = self.load_file(uuid)

        if self.transform:
            X = self.transform(X)

        X = pad_or_crop(X, self.window_size)
        assert X.size() == (self.window_size, 25)

        return X, Y # (T, 25)


class EgemapsFunctionalsDataset(Dataset):

    def __init__(self, config, uuids):
        self.config = config
        self.participant_ids = config.get('participant_ids')
        self.uuids = uuids
        self.window_size = np.rint(config.get('window_size_sec') * 30).astype(int) # fps
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.uuids)

    def get_transform(self):
        path = Path(self.config.get('db_processed')).parent / 'norm' / 'mean_std_egemaps_functionals.npz'
        if not path.exists(): return None
        mean, std = load_mean_std(path)
        return partial(standardization, mean=mean, std=std)

    def load_file(self, uuid):
        o = UUID(uuid)
        path = Path(self.config.get('db_processed')) / o.participant_id / 'egemaps' / f'{o.emotion_class}-{o.emotion_intensity}-{o.file_id}.npz'

        if not path.exists():
            raise FileNotFoundError(str(path))

        X = torch.FloatTensor(np.load(path)['functionals_full'])

        Y = {
            'emotion_class': torch.LongTensor([o.emotion_class]),
            'emotion_intensity': torch.LongTensor([o.emotion_intensity-1]),
            'file_id': o.file_id
        }

        return X, Y

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        X, Y = self.load_file(uuid)

        if self.transform:
            X = self.transform(X)

        X = X.unsqueeze(0).repeat(self.window_size, 1) # prepare sequence from this vector for training a sequence model..
        return X, Y # (N, 88)


class ClapDataset(Dataset):

    def __init__(self, config, uuids):
        self.config = config
        self.participant_ids = config.get('participant_ids')
        self.uuids = uuids
        self.window_size = np.rint(config.get('window_size_sec') * 30).astype(int) # fps

    def __len__(self):
        return len(self.uuids)

    def load_file(self, uuid):
        o = UUID(uuid)
        path = Path(self.config.get('db_processed')) / o.participant_id / 'clap' / f'{o.emotion_class}-{o.emotion_intensity}-{o.file_id}.npy'

        if not path.exists():
            raise FileNotFoundError(str(path))

        X = torch.FloatTensor(np.load(path))

        Y = {
            'emotion_class': torch.LongTensor([o.emotion_class]),
            'emotion_intensity': torch.LongTensor([o.emotion_intensity-1]),
            'file_id': o.file_id
        }

        return X, Y

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        X, Y = self.load_file(uuid)
        X = X.unsqueeze(0).repeat(self.window_size, 1) # prepare sequence from this vector for training a sequence model..
        return X, Y # (N, 1024)


class XmlRobertaDataset(Dataset):

    def __init__(self, config, uuids):
        self.config = config
        self.participant_ids = config.get('participant_ids')
        self.uuids = uuids
        self.window_size = np.rint(config.get('window_size_sec') * 30).astype(int) # fps

    def __len__(self):
        return len(self.uuids)

    def load_file(self, uuid):
        o = UUID(uuid)
        path = Path(self.config.get('db_processed')) / o.participant_id / 'xml_roberta' / f'{o.emotion_class}-{o.emotion_intensity}-{o.file_id}.npy'

        if not path.exists():
            raise FileNotFoundError(str(path))

        X = torch.FloatTensor(np.load(path))

        Y = {
            'emotion_class': torch.LongTensor([o.emotion_class]),
            'emotion_intensity': torch.LongTensor([o.emotion_intensity-1]),
            'file_id': o.file_id
        }

        return X, Y

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        X, Y = self.load_file(uuid)
        X = X.unsqueeze(0).repeat(self.window_size, 1) # prepare sequence from this vector for training a sequence model..
        return X, Y # (N, 768)


class MeadDataset(Dataset):

    def __init__(self, config: dict, uuids):
        self.config = config
        self.participant_ids = config.get('participant_ids')
        self.feature_list = config.get('feature_list')
        self.uuids = uuids
        self._setup_datasets(config, uuids)

    def _setup_datasets(self, config, uuids):
        self.datasets = {}
        for feature in self.feature_list:
            match feature:
                case "fabnet":
                    self.datasets[feature] = FabnetDataset(config, uuids)
                case "opengraphau":
                    self.datasets[feature] = OpenGraphAuDataset(config, uuids)
                case "egemaps_lld":
                    self.datasets[feature] = EgemapsLldDataset(config, uuids)
                case "egemaps_functionals":
                    self.datasets[feature] = EgemapsFunctionalsDataset(config, uuids)
                case "clap":
                    self.datasets[feature] = ClapDataset(config, uuids)
                case "xml_roberta":
                    self.datasets[feature] = XmlRobertaDataset(config, uuids)

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        features = {}
        labels = None
        for feature_name in self.feature_list:
            X, Y = self.datasets[feature_name][idx]
            features[feature_name] = X

            if labels is None:
                labels = Y

        return features, labels


def create_dataloaders(config: dict):
    test_uuids = generate_uuids(config.get('db_processed'), config.get('test_ids'))
    valid_uuids = generate_uuids(config.get('db_processed'), config.get('valid_ids'))
    train_uuids = generate_uuids(config.get('db_processed'), config.get('train_ids'))

    valid_uuids = valid_uuids[:int(round(len(valid_uuids) * config.get('data_percent', 100) / 100, ndigits=0))]
    train_uuids = train_uuids[:int(round(len(train_uuids) * config.get('data_percent', 100) / 100, ndigits=0))]

    test_dataset = MeadDataset(config, test_uuids)
    valid_dataset = MeadDataset(config, valid_uuids)
    train_dataset = MeadDataset(config, train_uuids)

    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32), shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.get('batch_size', 32), shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)

    return train_loader, valid_loader, test_loader


def calculate_standardization_params_opengraphau():
    print("Calculating standardization parameters for OpenGraphAU features...")
    config = load_config(Path("configs/dataloader_5i.yaml"))
    train_uuids = generate_uuids(config.get('db_processed'), config.get('train_ids'))
    train_dataset_au = OpenGraphAuDataset(config, train_uuids)
    train_loader_au = DataLoader(train_dataset_au, batch_size=config.get('batch_size', 100), shuffle=False, num_workers=8)
    mean, std = get_mean_std(train_loader_au, ndim=3) # (B, T, F)
    outpath = Path(config.get('db_processed')).parent / 'norm' / "mean_std_opengraphau"
    np.savez(str(outpath), mean=mean, std=std)
    print(f"OpenGraphAU standardization parameters are saved to {outpath}")


def calculate_standardization_params_egemaps_lld():
    print("Calculating standardization parameters for openSMILE eGeMAPS LLD features...")
    config = load_config(Path("configs/dataloader_5i.yaml"))
    train_uuids = generate_uuids(config.get('db_processed'), config.get('train_ids'))
    train_dataset_egemaps_lld = EgemapsLldDataset(config, train_uuids)
    train_loader_egemaps_lld = DataLoader(train_dataset_egemaps_lld, batch_size=config.get('batch_size', 100), shuffle=False, num_workers=8)
    mean, std = get_mean_std(train_loader_egemaps_lld, ndim=3) # (B, T, F)
    outpath = Path(config.get('db_processed')).parent / 'norm' / "mean_std_egemaps_lld"
    np.savez(str(outpath), mean=mean, std=std)
    print(f"eGeMAPS LLD standardization parameters are saved to {outpath}")


def calculate_standardization_params_egemaps_functionals():
    print("Calculating standardization parameters for openSMILE eGeMAPS Functionals features...")
    config = load_config(Path("configs/dataloader_5i.yaml"))
    train_uuids = generate_uuids(config.get('db_processed'), config.get('train_ids'))
    train_dataset_egemaps_functionals = EgemapsFunctionalsDataset(config, train_uuids)
    train_loader_egemaps_functionals = DataLoader(train_dataset_egemaps_functionals, batch_size=config.get('batch_size', 100), shuffle=False, num_workers=8)
    mean, std = get_mean_std(train_loader_egemaps_functionals, ndim=2) # (B, F)
    outpath = Path(config.get('db_processed')).parent / 'norm' / "mean_std_egemaps_functionals"
    np.savez(str(outpath), mean=mean, std=std)
    print(f"eGeMAPS Functionals standardization parameters are saved to {outpath}")


def check_data():
    config = load_config(Path("configs/dataloader_5i.yaml"))
    train_uuids = generate_uuids(config.get('db_processed'), config.get('train_ids'))
    train_dataset_egemaps_lld = EgemapsLldDataset(config, train_uuids)

    def contains_nan(tensor: torch.Tensor) -> bool:
        return torch.isnan(tensor).any().item()

    for x, y in tqdm(iter(train_dataset_egemaps_lld), total=len(train_dataset_egemaps_lld)):
        if contains_nan(x):
            print(x)
            print(y['file_id'])
            exit()


class MeadDataModule(pl.LightningDataModule):

    def __init__(self, config: dict | str):
        super().__init__()

        if isinstance(config, str) and Path(config).exists():
            config = load_config(config)

        self.config = config
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        test_uuids  = generate_uuids(self.config.get('db_processed'), self.config.get('test_ids'))
        valid_uuids = generate_uuids(self.config.get('db_processed'), self.config.get('valid_ids'))
        train_uuids = generate_uuids(self.config.get('db_processed'), self.config.get('train_ids'))
        # test_uuids  = train_uuids[:int(round(len(test_uuids) * self.config.get('data_percent', 100) / 100, ndigits=0))]
        valid_uuids = valid_uuids[:int(round(len(valid_uuids) * self.config.get('data_percent', 100) / 100, ndigits=0))]
        train_uuids = train_uuids[:int(round(len(train_uuids) * self.config.get('data_percent', 100) / 100, ndigits=0))]
        self.test_dataset  = MeadDataset(self.config, test_uuids)
        self.valid_dataset = MeadDataset(self.config, valid_uuids)
        self.train_dataset = MeadDataset(self.config, train_uuids)
        return self

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.get('batch_size', 32),
                          shuffle=True,
                          num_workers=psutil.cpu_count(logical=False),
                          persistent_workers=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.config.get('batch_size', 32),
                          shuffle=False,
                          num_workers=psutil.cpu_count(logical=False),
                          persistent_workers=True,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.config.get('batch_size', 32),
                          shuffle=False,
                          num_workers=psutil.cpu_count(logical=False),
                          persistent_workers=True,
                          pin_memory=True)

    def setup_and_test(self):
        test_uuids = generate_uuids(self.config.get('db_processed'), self.config.get('test_ids'))
        test_dataset = MeadDataset(self.config, test_uuids)
        return DataLoader(test_dataset,
                          batch_size=self.config.get('batch_size', 32),
                          shuffle=False,
                          num_workers=psutil.cpu_count(logical=False),
                          persistent_workers=True,
                          pin_memory=True)

    def setup_and_test_camera(self, camera_angle: str):
        test_uuids = generate_uuids(self.config.get('db_processed'), self.config.get('test_ids'))
        test_uuids = [uuid for uuid in test_uuids if uuid.split("-")[2] == camera_angle]
        test_dataset = MeadDataset(self.config, test_uuids)
        return DataLoader(test_dataset,
                          batch_size=self.config.get('batch_size', 32),
                          shuffle=False,
                          num_workers=psutil.cpu_count(logical=False),
                          persistent_workers=True,
                          pin_memory=True)

    def print_stats(self):
        print('[train dataloader]', '\n\tnumber of batches:', len(self.train_dataloader()))
        print('[valid dataloader]', '\n\tnumber of batches:', len(self.val_dataloader()))
        print('[test dataloader]', '\n\tnumber of batches:', len(self.test_dataloader()))


if __name__ == "__main__":
    # calculate_standardization_params_opengraphau()
    # calculate_standardization_params_egemaps_lld()
    calculate_standardization_params_egemaps_functionals()
    exit()

    config = load_config(Path("data/db_processed/configs/advanced.yaml"))
    train_loader, valid_loader, test_loader = create_dataloaders(config)
    for x, y in tqdm(test_loader, total=len(test_loader)):
        print(y['emotion_class'], y['emotion_intensity'])
        assert not torch.isnan(x['egemaps_lld']).any()
        assert not torch.isnan(x['fabnet']).any()
        assert not torch.isnan(x['opengraphau']).any()
        assert not torch.isnan(x['xml_roberta']).any()
        assert not torch.isnan(x['clap']).any()

    for x, y in tqdm(valid_loader, total=len(valid_loader)):
        print(y['emotion_class'], y['emotion_intensity'])
        assert not torch.isnan(x['egemaps_lld']).any()
        assert not torch.isnan(x['fabnet']).any()
        assert not torch.isnan(x['opengraphau']).any()
        assert not torch.isnan(x['xml_roberta']).any()
        assert not torch.isnan(x['clap']).any()

    for x, y in tqdm(train_loader, total=len(train_loader)):
        print(y['emotion_class'], y['emotion_intensity'])
        assert not torch.isnan(x['egemaps_lld']).any()
        assert not torch.isnan(x['fabnet']).any()
        assert not torch.isnan(x['opengraphau']).any()
        assert not torch.isnan(x['xml_roberta']).any()
        assert not torch.isnan(x['clap']).any()