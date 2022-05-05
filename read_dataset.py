import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import h5py


class NinaProDB8:
    def __init__(self, path, restimulus=3, glove_motor=5, train=True, use_amputee=False):
        assert os.path.exists(path)
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".mat")]
        if not use_amputee:
            self.files = [f for f in self.files if "S11" not in f and "S12" not in f]

        if train:
            self.files = [f for f in self.files if "A3" not in f]
        else:
            self.files = [f for f in self.files if "A3" in f]
        self.restimulus = restimulus
        self.glove_motor = glove_motor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, id):
        file = self.files[id]
        dat = loadmat(file)
        mask = (dat["restimulus"] == self.restimulus).flatten()

        emg_dat = dat["emg"][mask, :]
        glove_dat = dat["glove"][mask, self.glove_motor]

        return emg_dat, glove_dat


class NinaProDB8Lite:
    def __init__(self, path, train=True):
        self.path = os.path.join(path, "train.h5" if train else "test.h5")
        with h5py.File(self.path, "r") as f:
            self.ids = list(f.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, id):
        with h5py.File(self.path, "r") as f:
            grp = f[self.ids[id]]
            return grp["emg"][:], grp["glove"][:]


if __name__ == "__main__":
    """Creates the DB8 Lite dataset from src dataset at location set by NINA_DATASET_PATH env variable"""
    src_path = os.environ.get("NINA_DATASET_PATH")
    target_path = "dataset"
    os.makedirs(target_path, exist_ok=True)
    train_target = h5py.File(os.path.join(target_path, "train.h5"), "w")
    test_target = h5py.File(os.path.join(target_path, "test.h5"), "w")

    train_set = NinaProDB8(src_path)
    test_set = NinaProDB8(src_path, train=False)

    for i in range(len(train_set)):
        try:
            (emg_dat, glove_dat) = train_set[i]
        except:
            continue
        grp = train_target.create_group(Path(train_set.files[i]).stem)
        grp["emg"] = emg_dat
        grp["glove"] = glove_dat

    train_target.close()

    for i in range(len(test_set)):
        try:
            (emg_dat, glove_dat) = test_set[i]
        except:
            continue
        grp = test_target.create_group(Path(train_set.files[i]).stem)
        grp["emg"] = emg_dat
        grp["glove"] = glove_dat

    test_target.close()
