import os
import scipy.io
from metadata import subject_dataset
from src.utils import *
from src.SubjectSpikeModel import *

# from src.ADM import digitalize_sigma_J
# from src.ADM import ADM

pd.set_option('display.max_rows', 1000)

data_dir = 'data/'
subject_file = f'S{subject}_E1_A{subject_dataset}.mat'  # S1_E1_A1.mat

logging.basicConfig(level=logging.INFO)

V_THR = 0.3e-5
T_REF = 0.0001
ADM_DT = 1 / SAMP_FREQ


def main():
    mat = scipy.io.loadmat(os.path.join(data_dir, subject_file))
    rec_time, n_ch, n_rep, n_stim = extract_rec_info(mat)
    logging.info(f"rec time:{rec_time}   n_ch:{n_ch}   n_stim:{n_stim}  n_rep:{n_rep}")
    emg_df = construct_emg_df(mat, rec_time)
    glove_df = construct_glove_df(emg_df, mat, rec_time)

    # extract channels only
    X = emg_df.iloc[:, :n_ch].to_numpy()
    y = glove_df.to_numpy()
    print(f"X (EMG dataset) shape:{X.shape}   target:{y.shape}")

    # test on subset
    test_subset(X)

    # loop over trials + channels to populate the class  # trial_reconst, spike_up, spike_dn, spike_time_up,  
    # spike_time_dn = digitalize_sigma_J(trial_data, V_THR, T_REF,  #  # ADM_DT)

    # dataset slicing
    stim_spikes_model = {}
    for stim in n_stim:
        reps = emg_df[(emg_df['restimulus'] == stim)]['rerepetition'].unique()
        list_rep_model = []
        for rep in reps:
            list_ch_model = []
            trial_df = emg_df[(emg_df['restimulus'] == stim) & (emg_df['rerepetition'] == rep)]
            trial_data = trial_df.iloc[:100, :n_ch].to_numpy()
            logging.debug(f"stim#{stim}   rep:{rep}  trialdf:{trial_data.shape}")
            for ch in range(n_ch):
                trial_reconst, spike_up, spike_dn, spike_time_up, spike_time_dn = digitalize_sigma_J(trial_data[:, ch],
                                                                                                     V_THR, T_REF,
                                                                                                     ADM_DT)
                sp_id_up = np.repeat(ch, len(spike_time_up))
                sp_id_dn = np.repeat(ch + n_ch, len(spike_time_dn))
                list_ch_model.append(ChannelSpikesModel(spike_time_up, sp_id_up, spike_time_dn, sp_id_dn))

            list_rep_model.append(TrialSpikesModel(list_ch_model))
        stim_spikes_model[stim] = list_rep_model

    # testing
    stim_test = 2
    rep_test = 1
    print(f"Getting trial {rep_test} stim {stim_test} spikes:\n{stim_spikes_model[stim_test][rep_test].trial_times}")


def test_subset(X):
    ch_test = 0
    trial_data = X[:200, ch_test]
    time_test = np.arange(0, 0.1, 1 / SAMP_FREQ)
    trial_reconst, spike_up, spike_dn, spike_time_up, spike_time_dn = digitalize_sigma_J(trial_data, V_THR, T_REF,
                                                                                         ADM_DT)
    plot_spikes(trial_data, trial_reconst, time_test, spike_time_dn, spike_time_up)


if __name__ == "__main__":
    main()
