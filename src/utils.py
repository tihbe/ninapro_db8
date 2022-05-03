# import seaborn as sns
from scipy import signal
from tqdm import tqdm
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import numpy as np
import logging
import pandas as pd
from metadata import SAMP_FREQ, subject, MOV_DUR_IN_SEC, COLOR_DICT
import matplotlib.pyplot as plt
from ADM import digitalize_sigma_J


# import seaborn as sns


def construct_glove_df(emg_df, mat, rec_time):
    glove_df = pd.DataFrame(mat['glove'])
    # glove_df['time'] = rec_time
    # glove_df['restimulus'] = mat['restimulus']
    # glove_df['rerepetition'] = mat['rerepetition']
    # glove_df['trial'] = emg_df['trial']
    # logging.debug(f"Checking on unique restimulus {glove_df[glove_df['rerepetition'] == 2]['restimulus'].unique()}")
    # logging.debug(f"Checking on id of first repetition per restimulus:\n "
    #               f"{glove_df.groupby('restimulus')['rerepetition'].first()}")
    return glove_df


def construct_emg_df(mat, rec_time):
    emg_df = pd.DataFrame(mat['emg'])
    emg_df['time'] = rec_time
    emg_df['restimulus'] = mat['restimulus']
    emg_df['rerepetition'] = mat['rerepetition']
    emg_df['trial'] = (emg_df['restimulus'] != emg_df['restimulus'].shift()).cumsum()
    return emg_df


def extract_rec_info(mat):
    n_ch = mat['emg'].shape[1]
    rec_duration = mat['emg'].shape[0] / SAMP_FREQ
    rec_time = np.arange(0, rec_duration, 1 / SAMP_FREQ)
    logging.info(f"Total recording duration for this subject {subject}: {rec_duration} seconds")
    logging.info(f"Creating time axis: \nbeg {rec_time[:5]} \nend {rec_time[-5:]}\n")
    logging.info(f"sEMG Channels: {n_ch}")
    return rec_time, n_ch


def check_sp_ref_period(t_up, t_dn, t_ref):
    isi_up = np.diff(t_up)
    isi_dn = np.diff(t_dn)
    assert np.any(isi_up < t_ref), "t_ref between the encoded UP spikes is note respected!"
    assert np.any(isi_dn < t_ref), "t_ref between the encoded DN spikes is note respected!"

    # if np.any(isi_up < T_REF):  #     min_isi_loc = np.argwhere(isi_up < T_REF).flatten()  #     first_sp =  #  #
    # np.asarray(t_up)[min_isi_loc]  #     second_sp = np.asarray(t_up)[min_isi_loc + 1]  #     print("ISI:\n{  #  #
    # }\n".format(second_sp - first_sp))  # if np.any(isi_dn < T_REF):  #     min_isi_loc = np.argwhere(isi_dn <  #
    # T_REF).flatten()  #     first_sp = np.asarray(t_dn)[min_isi_loc]  #     second_sp = np.asarray(t_dn)[  #  #  #
    # min_isi_loc + 1]  #     print("ISI:\n{}\n".format(second_sp - first_sp))


def plot_spikes(X, reconst, time, dn_sp, up_sp):
    plt.figure()
    plt.plot(time, X, 'o-', label='original', color=COLOR_DICT['midnight_blue'])
    plt.plot(time, reconst, 'o-', label='reconstructed', color=COLOR_DICT['pumpkin'])
    plt.plot(up_sp, np.repeat(np.max(X), len(up_sp)), '|', markersize=10, color=COLOR_DICT['green_sea'])
    plt.plot(dn_sp, np.repeat(np.max(X), len(dn_sp)), '|', markersize=10, color=COLOR_DICT['pomgrenate'])

    # plt.title(f'Ch #{pca_ch} (ch_id {ch_id}), Trial: {trial + 1}, Gesture:{gesture}   UP:{ADM_V_THR}  tref: '
    #           f'{ADM_T_REF}')
    plt.xlabel('time cropped to 200ms window [s]')
    plt.ylabel(r'Amplitude $\mu V$')
    plt.legend()
    plt.show()

    # plt.savefig(RESULTDIR + 'spike_plots/' + ENG_KEY +'/' + str(RESAMPLE_FREQUENCY) + 'Hz/' + title +  # '.png')
