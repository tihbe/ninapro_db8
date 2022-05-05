from read_dataset import NinaProDB8Lite
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy.signal

train_set = NinaProDB8Lite("dataset")

sampling_frequency = 2000

for i, (emg, glove) in enumerate(train_set):
    emg_duration, nb_channels = emg.shape
    glove_gaussian_filtered = gaussian_filter1d(glove, sigma=500)

    glove_gaussian_filtered = glove_gaussian_filtered / glove_gaussian_filtered.max() * 255
    glove_gaussian_filtered = glove_gaussian_filtered.astype(np.uint8)

    env_emg = np.abs(scipy.signal.hilbert(emg, axis=0))

    # Custom a-LIF encoding
    mem_pot = np.zeros(nb_channels)
    dt = 1 / sampling_frequency
    tau = 20e-3

    th_rest = np.max(env_emg)
    th = th_rest * np.ones(nb_channels)

    th_inc = env_emg.mean()
    th_tau = 0.500

    spikes = np.empty((emg_duration, nb_channels), dtype=np.int8)

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

    fig, axs = plt.subplots(2, sharex=True)
    axs[0].eventplot([np.flatnonzero(i) for i in spikes.T])
    axs[1].plot(glove_gaussian_filtered)
    plt.show()
