import os
import h5py
from read_dataset import NinaProDB8Lite
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy.signal
from tqdm import tqdm


def main():
    encode("dataset", True)
    encode("dataset", False)


class NinaProDB8ALIFEncoded:
    def __init__(self, path, train=True):
        self.path = os.path.join(path, "train_spikes.h5" if train else "test_spikes.h5")
        with h5py.File(self.path, "r") as f:
            self.ids = list(f.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, id):
        with h5py.File(self.path, "r") as f:
            grp = f[self.ids[id]]
            return grp["spikes"][:], grp["glove"][:]


def encode(path, train):
    sampling_frequency = 2000
    data_set = NinaProDB8Lite(path, train=train)
    type_str = "train" if train else "test"
    target = h5py.File(os.path.join(path, f"{type_str}_spikes.h5"), "w")

    for i, (emg, glove) in enumerate(tqdm(data_set)):

        glove_gaussian_filtered = gaussian_filter1d(glove, sigma=500)

        # Best attempt at removing plateaus
        plateaus = np.isclose(np.diff(glove_gaussian_filtered), 0, atol=1e-3)
        emg = emg[:-1, :][~plateaus, :]
        glove_gaussian_filtered = glove_gaussian_filtered[:-1][~plateaus]

        # Convert glove to uint8
        glove_gaussian_filtered = glove_gaussian_filtered / glove_gaussian_filtered.max() * 255
        glove_gaussian_filtered = glove_gaussian_filtered.astype(np.uint8)

        env_emg = np.abs(scipy.signal.hilbert(emg, axis=0))

        emg_duration, nb_channels = emg.shape

        # Custom a-LIF encoding
        mem_pot = np.zeros(nb_channels)
        dt = 1 / sampling_frequency
        tau = 20e-3

        th_rest = np.max(env_emg)
        th = th_rest * np.ones(nb_channels)

        th_inc = env_emg.mean()
        th_tau = 0.500

        spikes = np.empty((emg_duration, nb_channels), dtype=np.uint8)

        cst = np.exp(-dt / tau)
        cst_th = np.exp(-dt / th_tau)

        # Go through the data once to figure out the threshold with A-LIF model
        for t in range(emg_duration):
            mem_pot = mem_pot * cst + env_emg[t]
            th = (th - th_rest) * cst + th_rest
            spikes[t] = mem_pot > th
            mem_pot *= 1 - spikes[t]
            th += spikes[t] * th_inc

        # Use fixed threshold to actually encode the data with LIF model
        for t in range(emg_duration):
            mem_pot = mem_pot * cst + env_emg[t]
            spikes[t] = mem_pot > th
            mem_pot *= 1 - spikes[t]

        grp = target.create_group(data_set.ids[i])
        grp["spikes"] = spikes
        grp["glove"] = glove_gaussian_filtered

        if show_plots := False:
            fig, axs = plt.subplots(2, sharex=True)
            axs[0].eventplot([np.flatnonzero(i) for i in spikes.T])
            axs[1].plot(glove_gaussian_filtered)
            plt.show()


if __name__ == "__main__":
    main()
