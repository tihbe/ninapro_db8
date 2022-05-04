import os
import scipy.io
from metadata import SUBJECT_FILE, DATA_DIR
from src.utils import *

pd.set_option('display.max_rows', 1000)

logging.basicConfig(level=logging.INFO)

# ADM parameters
V_THR = 0.5e-5
T_REF = 0.01
ADM_DT = 1 / SAMP_FREQ


def main():
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, SUBJECT_FILE))
    rec_time, n_ch, n_rep, n_stim = extract_rec_info(mat)
    logging.info(f"rec time:{rec_time}   n_ch:{n_ch}   n_stim:{n_stim}  n_rep:{n_rep}")
    emg_df = construct_emg_df(mat, rec_time)
    glove_df = construct_glove_df(mat)

    # extract channels only
    X = emg_df.iloc[:, :n_ch].to_numpy()
    y = glove_df.to_numpy()
    print(f"X (EMG dataset) shape:{X.shape}   target:{y.shape}")

    # ignore/comment if you are using a different encoding than ADM
    stim_spikes_model = encode_into_spikes(emg_df, n_ch, n_stim, V_THR, T_REF, ADM_DT)
    # Quick Sanity Check
    stim_test = 2
    rep_test = 1
    logging.debug(
        f"Getting trial {rep_test} stim {stim_test} spikes:\n{stim_spikes_model[stim_test][rep_test].trial_times}")


if __name__ == "__main__":
    main()
