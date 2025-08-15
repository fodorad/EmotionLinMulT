import webdataset as wds
import h5py
import numpy as np
from pathlib import Path

import sys
import os
sys.path.insert(0, '/home/fodorad/Dev/emotionlinmult')

from emotionlinmult.preprocess.AFEWVA.dataset_face import create_dataset_with_size as create_dataset_afewva
from emotionlinmult.preprocess.RAVDESS.dataset import create_dataset_with_size as create_dataset_ravdess
from emotionlinmult.preprocess.MOSEI.dataset import create_dataset_with_size as create_dataset_mosei
from emotionlinmult.preprocess.CREMAD.dataset_expr import create_dataset_with_size as create_dataset_cremad_expr
from emotionlinmult.preprocess.CREMAD.dataset_int import create_dataset_with_size as create_dataset_cremad_int
from emotionlinmult.preprocess.AffWild2.dataset_expr import create_dataset_with_size as create_dataset_affwild2_expr
from emotionlinmult.preprocess.AffWild2.dataset_va import create_dataset_with_size as create_dataset_affwild2_va

def save_webdataset_to_hdf5(datasets_dict, h5_path):
    """
    Save multiple subsets of datasets with dynamic keys.

    Args:
        datasets_dict: dict of subset_name -> dataset iterable
            Each dataset yields sample dicts containing any number of keys.
            Keys ending with '_mask.npy' are boolean masks.
            Other tensor keys are converted using .numpy().
            String keys are saved as variable length UTF-8 strings.
        h5_path: output HDF5 file path.
    """
    str_dt = h5py.string_dtype(encoding='utf-8')

    with h5py.File(h5_path, 'w') as h5file:
        for subset_name, dataset in datasets_dict.items():
            print(f"[{Path(h5_path).stem}] Saving subset '{subset_name}'")

            group = h5file.create_group(subset_name)
            created = False
            dataset_keys = []  # to keep track of which datasets we created

            for i, sample in enumerate(dataset):
                # On first iteration, create datasets dynamically to match sample keys
                if not created:
                    for key, val in sample.items():
                        # Detect if string values
                        if isinstance(val, (str, bytes)):
                            # create string dataset
                            group.create_dataset(
                                key if key not in ['__key__', 'dataset'] else
                                ('keys' if key == '__key__' else 'datasets'),
                                shape=(1,), maxshape=(None,), dtype=str_dt, chunks=True
                            )
                            dataset_keys.append(key)
                        else:
                            # Assume tensor or numpy array, convert accordingly
                            if hasattr(val, 'numpy'):
                                np_val = val.numpy()
                            else:
                                np_val = np.array(val)

                            if key.endswith('_mask.npy'):
                                dtype = np.bool_
                            else:
                                dtype = np_val.dtype

                            # create resizable dataset
                            group.create_dataset(
                                key if key not in ['__key__', 'dataset'] else
                                ('keys' if key == '__key__' else 'datasets'),
                                shape=(1,) + np_val.shape,
                                maxshape=(None,) + np_val.shape,
                                dtype=dtype,
                                chunks=True
                            )
                            dataset_keys.append(key)

                    created = True

                # Resize each dataset by 1 on axis 0
                for key in dataset_keys:
                    dset_name = key if key not in ['__key__', 'dataset'] else \
                               ('keys' if key == '__key__' else 'datasets')
                    dset = group[dset_name]
                    dset.resize(dset.shape[0] + 1, axis=0)

                # Write data for this sample
                for key, val in sample.items():
                    dset_name = key if key not in ['__key__', 'dataset'] else \
                               ('keys' if key == '__key__' else 'datasets')

                    if isinstance(val, (str, bytes)):
                        group[dset_name][-1] = val
                    else:
                        if hasattr(val, 'numpy'):
                            arr = val.numpy()
                        else:
                            arr = np.array(val)
                        if key.endswith('_mask.npy'):
                            arr = arr.astype(bool)
                        group[dset_name][-1, ...] = arr

                if i % 100 == 0:
                    print(f"  [{Path(h5_path).stem}] Saved {i+1} samples in subset '{subset_name}'")

            print(f"[{Path(h5_path).stem}] Finished saving {i+1} samples in subset '{subset_name}'")

    print(f"[{Path(h5_path).stem}] All subsets saved dynamically to {h5_path}")


output_dir = Path("/media/fodorad/Kinga HDD/db_processed")
"""
datasets_afewva = {
    "train": create_dataset_afewva('train'),
    "valid": create_dataset_afewva('valid'),
    "test": create_dataset_afewva('test'),
}

save_webdataset_to_hdf5(datasets_afewva, output_dir / "AFEWVA.h5")


datasets_ravdess = {
    "train": create_dataset_ravdess('train'),
    "valid": create_dataset_ravdess('valid'),
    "test": create_dataset_ravdess('test')
}

save_webdataset_to_hdf5(datasets_ravdess, output_dir / "RAVDESS.h5")


datasets_cremad_expr = {
    "train": create_dataset_cremad_expr('train'),
    "valid": create_dataset_cremad_expr('valid'),
    "test": create_dataset_cremad_expr('test')
}

save_webdataset_to_hdf5(datasets_cremad_expr, output_dir / "CREMAD_EXPR.h5")
"""

datasets_cremad_int = {
    "train": create_dataset_cremad_int('train'),
    "valid": create_dataset_cremad_int('valid'),
    "test": create_dataset_cremad_int('test')
}

save_webdataset_to_hdf5(datasets_cremad_int, output_dir / "CREMAD_INT.h5")
"""

datasets_affwild2_expr = {
    "train": create_dataset_affwild2_expr('train'),
    "valid": create_dataset_affwild2_expr('valid'),
    "test": create_dataset_affwild2_expr('test')
}

save_webdataset_to_hdf5(datasets_affwild2_expr, output_dir / "AFFWILD2_EXPR.h5")


datasets_affwild2_va = {
    "train": create_dataset_affwild2_va('train'),
    "valid": create_dataset_affwild2_va('valid'),
    "test": create_dataset_affwild2_va('test')
}

save_webdataset_to_hdf5(datasets_affwild2_va, output_dir / "AFFWILD2_VA.h5")

datasets_mosei = {
    "train": create_dataset_mosei('train'),
    "valid": create_dataset_mosei('valid'),
    "test": create_dataset_mosei('test')
}

save_webdataset_to_hdf5(datasets_mosei, output_dir / "MOSEI.h5")
"""