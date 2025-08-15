import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


class HDF5SubsetDataset(Dataset):
    def __init__(self, h5_path, subset_name, keys=None):
        """
        Args:
            h5_path (str or Path): Path to the HDF5 file.
            subset_name (str): One of 'train', 'valid', 'test', etc. The group name in the file.
            keys (list[str] or None): Only load these specific keys. If None, load all keys.
        """
        self.h5_path = Path(h5_path)
        self.subset_name = subset_name
        # It's best practice to open the file only in __getitem__ to avoid multi-process h5py issues
        with h5py.File(self.h5_path, 'r') as h5file:
            group = h5file[self.subset_name]
            # Collect keys if not specified
            if keys is None:
                self.keys = list(group.keys())
            else:
                self.keys = keys
            self.length = group[self.keys[0]].shape[0]  # Sample count
            self.dataset_shapes = {key: group[key].shape for key in self.keys}
            self.string_dtypes = [key for key in self.keys if group[key].dtype.kind in {"U", "S", "O"}]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Re-open the HDF5 file each time for thread/process safety
        with h5py.File(self.h5_path, 'r') as h5file:
            group = h5file[self.subset_name]
            sample = {}
            for key in self.keys:
                data = group[key][idx]
                # Convert strings to Python str
                if key in self.string_dtypes:
                    value = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                else:
                    value = torch.from_numpy(np.array(data))
                    if key.endswith("_mask.npy"):
                        value = value.bool()
                sample[key] = value
            return sample


def create_dataset(db_path: str, subset_name: str):
    if db_path is None:
        raise ValueError("Unsupported. DB path is None")
    return HDF5SubsetDataset(h5_path=db_path, subset_name=subset_name)


if __name__ == "__main__":

    import time
    db_paths = {
        "AFEWVA":        "data/db_processed/AFEW-VA/AFEWVA.h5",
        "RAVDESS":       "data/db_processed/RAVDESS/RAVDESS.h5",
        "MOSEI":         "data/db_processed/MOSEI/MOSEI.h5",
        "CREMAD_EXPR":   "data/db_processed/CREMA-D/CREMAD_EXPR.h5",
        "CREMAD_INT":    "data/db_processed/CREMA-D/CREMAD_INT.h5",
        "AFFWILD2_EXPR": "data/db_processed/Aff-Wild2/AFFWILD2_EXPR.h5",
        "AFFWILD2_VA":   "data/db_processed/Aff-Wild2/AFFWILD2_VA.h5"
    }

    for db_name, db_path in db_paths.items():
        ds = create_dataset(db_path, "train")
        dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
        for batch_ind, batch in enumerate(tqdm(dl, total=len(dl), desc=f"{db_name}")):
            print(db_name, batch.keys())
            assert batch['datasets'] == [batch['datasets'][0]] * len(batch['datasets'])
    
    exit()

    for db_name, db_path in db_paths.items():
        for subset_name in ["train", "valid", "test"]:
            ds = create_dataset(db_path, subset_name)
            dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=10, persistent_workers=True)
            start_time = time.time()
            for batch_ind, batch in enumerate(tqdm(dl, total=len(dl), desc=f"{db_name} {subset_name}")):
                tensors = [batch[key].to(device="mps") for key in batch if key.endswith(".npy")]
            print(f"{db_name} {subset_name} time: {time.time() - start_time}")
