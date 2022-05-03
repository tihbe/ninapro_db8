import numpy as np  # Version when written 1.19.2
from numba import njit  # Version when written 0.51.2
from scipy.interpolate import interp1d
from metadata import MOV_DUR_IN_SEC, SAMP_FREQ
from src.SpikeModels import *
from utils import *


#
# def ADM(input_signal, threshold_UP, threshold_DOWN, sampling_frequency, refractory_period_duration,
# return_indices=True,
#         index_dt=1e-4):
#     dt = 1 / sampling_frequency
#     end_time = len(input_signal) * dt
#     times = np.arange(0, end_time, dt)
#     if not refractory_period_duration:
#         spike_t_up, spike_t_dn, times_interpolated, spike_idx_up, spike_idx_dn = ADM_numba_noRFR(input_signal,
#                                                                                                  threshold_UP,
#                                                                                                  threshold_DOWN,
#                                                                                                  sampling_frequency,
#                                                                                                  return_indices,
#                                                                                                  index_dt)
#
#     else:
#         if refractory_period_duration < dt:
#             interpolation_factor = 1
#             while dt > refractory_period_duration:
#                 interpolation_factor += 1
#                 dt = 1 / (sampling_frequency * interpolation_factor)
#             f = interp1d(times, input_signal)
#             times = np.concatenate((np.arange(0, times[-1], dt), [times[-1]]))
#             input_signal = f(times)
#             sampling_frequency = 1 / times[1]
#         spike_t_up, spike_t_dn, times_interpolated, spike_idx_up, spike_idx_dn = ADM_numba(input_signal, threshold_UP,
#                                                                                            threshold_DOWN,
#                                                                                            sampling_frequency,
#                                                                                            refractory_period_duration,
#                                                                                            return_indices, index_dt)
#     return (spike_t_up, spike_t_dn, times_interpolated, spike_idx_up, spike_idx_dn)
#
#
# @njit(fastmath=True, parallel=True)
# def ADM_numba(input_signal, threshold_UP, threshold_DOWN, sampling_frequency, refractory_period_duration,
#               return_indices, index_dt):
#     dt = 1 / sampling_frequency
#     end_time = len(input_signal) * dt
#     times = np.linspace(0, end_time, len(input_signal)).astype(np.float64)
#     DC_Voltage = input_signal[0]
#     remainder_of_refractory = 0
#     spike_t_up = times[0:2]
#     spike_t_dn = times[0:2]
#     interpolate_from = 0.0
#     interpolation_activation = 0
#     intercept_point = 0
#
#     for i in range(len(times)):
#         t = i * dt
#         if i == 0:
#             continue
#
#         slope = ((input_signal[i] - input_signal[i - 1]) / dt)
#         if remainder_of_refractory >= 2 * dt:
#             remainder_of_refractory = remainder_of_refractory - dt
#             interpolation_activation = 1
#
#         else:
#
#             if interpolation_activation == 1:
#                 interpolate_from = (interpolate_from + remainder_of_refractory)
#                 remainder_of_refractory = 0
#                 if interpolate_from >= 2 * dt:
#                     interpolate_from = interpolate_from - dt
#                     continue
#                 interpolate_from = (interpolate_from + remainder_of_refractory) % dt
#                 Vbelow = (input_signal[i - 1] + interpolate_from * slope)
#                 DC_Voltage = Vbelow
#
#
#             else:
#                 Vbelow = input_signal[i - 1]
#                 interpolate_from = 0
#
#             if DC_Voltage + threshold_UP <= input_signal[i]:
#                 intercept_point = t - dt + interpolate_from + ((threshold_UP + DC_Voltage - Vbelow) / slope)
#                 spike_t_up = np.append(spike_t_up, intercept_point)
#                 interpolate_from = dt + intercept_point - t
#                 remainder_of_refractory = refractory_period_duration
#                 interpolation_activation = 1
#                 continue
#
#             elif DC_Voltage - threshold_DOWN >= input_signal[i]:
#                 intercept_point = t - dt + interpolate_from + ((-threshold_DOWN + DC_Voltage - Vbelow) / slope)
#                 spike_t_dn = np.append(spike_t_dn, intercept_point)
#                 interpolate_from = dt + intercept_point - t
#                 remainder_of_refractory = refractory_period_duration
#                 interpolation_activation = 1
#                 continue
#
#             interpolation_activation = 0
#
#     index = [0, 1]
#     spike_t_up = np.delete(spike_t_up, index)
#     spike_t_dn = np.delete(spike_t_dn, index)
#
#     if return_indices:
#         times_interpolated = np.arange(0, end_time, index_dt)
#         spike_idx_up = np.zeros_like(times_interpolated)
#         spike_idx_dn = np.zeros_like(times_interpolated)
#         idxdn = np.searchsorted(times_interpolated, spike_t_dn)
#         spike_idx_dn[idxdn] = 1
#         idxup = np.searchsorted(times_interpolated, spike_t_up)
#         spike_idx_up[idxup] = 1
#     return (spike_t_up, spike_t_dn, times_interpolated, spike_idx_up, spike_idx_dn)
#
#
# @njit(fastmath=True, parallel=True)
# def ADM_numba_noRFR(input_signal, threshold_UP, threshold_DOWN, sampling_frequency, return_indices, index_dt):
#     dt = 1 / sampling_frequency
#     end_time = len(input_signal) * dt
#     times = np.linspace(0, end_time, len(input_signal)).astype(np.float64)
#     DC_Voltage = input_signal[0]
#     spike_t_up = times[0:2]
#     spike_t_dn = times[0:2]
#     intercept_point = 0
#     for i in range(len(times)):
#         t = i * dt
#         if i == 0:
#             continue
#
#         slope = ((input_signal[i] - input_signal[i - 1]) / dt)
#
#         V_above = input_signal[i]
#         V_below = input_signal[i - 1]
#
#         if V_above >= DC_Voltage + threshold_UP:
#             while V_above >= DC_Voltage + threshold_UP:
#                 intercept_point = t - dt + ((threshold_UP + DC_Voltage - V_below) / slope)
#                 spike_t_up = np.append(spike_t_up, intercept_point)
#                 DC_Voltage = DC_Voltage + threshold_UP
#
#         elif V_above <= DC_Voltage - threshold_DOWN:
#             while V_above <= DC_Voltage - threshold_DOWN:
#                 intercept_point = t - dt + ((-threshold_DOWN + DC_Voltage - V_below) / slope)
#                 spike_t_dn = np.append(spike_t_dn, intercept_point)
#                 DC_Voltage = DC_Voltage - threshold_DOWN
#
#     index = [0, 1]
#     spike_t_up = np.delete(spike_t_up, index)
#     spike_t_dn = np.delete(spike_t_dn, index)
#
#     if return_indices:
#         times_interpolated = np.arange(0, end_time, index_dt)
#         spike_idx_up = np.zeros_like(times_interpolated)
#         spike_idx_dn = np.zeros_like(times_interpolated)
#         idxdn = np.searchsorted(times_interpolated, spike_t_dn)
#         spike_idx_dn[idxdn] = 1
#         idxup = np.searchsorted(times_interpolated, spike_t_up)
#         spike_idx_up[idxup] = 1
#     return (spike_t_up, spike_t_dn, times_interpolated, spike_idx_up, spike_idx_dn)
#
#
class Line:
    '''straight line that goes through points:
        p1: (i*dt, x1)
        p2: ((i+1)*dt, x2)
    '''

    def __init__(self, i, dt, x1, x2):
        self.m = (x2 - x1) / dt
        self.n = (i + 1) * x1 - i * x2
        self.x1 = x1
        self.x2 = x2

    def f(self, t):
        ''' straight line through (t1,x1) and (t2,x2): f(t) = t*m + n'''
        return self.m * t + self.n

    def g(self, x):
        ''' inverse of straight line fx: g(x) = (x-n)/m. Assumes m!=0'''
        return (x - self.n) / self.m

    def in_range(self, x, v_thr):
        ''' checks if x±v_thr is in [x1,x2] or [x2,x1] '''
        mn = min(self.x1, self.x2)
        mx = max(self.x1, self.x2)
        if (mn <= x + v_thr < mx):
            return 1
        elif (mn <= x - v_thr < mx):
            return -1
        else:
            return 0


def digitalize_sigma_J(x_in, v_thr, t_ref, dt):
    '''
    Sigma modulation of input signal x, implicitly assuming a linear interpolation
    between the points of x (as "x" was a piecewise defined function, composed
    of straight lines). Considers only one voltage threshold for both increasing
    and decreasing x.
    :x (float array): input signal.
    :t_ref (float): refractory period after a spike (no other spikes generated).
    :v_thr (float): threshold that generates a spike when signal goes above or
                    below it
    :return y (float array): quantized signal
            spike_up number of up spikes at each time point
            spike_dn number of down spikes at each time point
    '''
    y = np.zeros(len(x_in))
    y[0] = x_in[0]
    t1 = 0  # when is error signal starting to count (t_spike + t_refractory)
    x1 = x_in[0]

    spike_up = []
    spike_dn = []
    spike_time_up = []
    spike_time_dn = []

    for i in range(len(x_in) - 1):
        # if i > 50:
        #     lol
        if t1 < (i + 1) * dt:
            # enter here if a spike could occur because t_ref has passed
            x = Line(i, dt, x_in[i], x_in[i + 1])  # continuous version of x_in
            if not x1:
                x1 = x.f(t1)
            ir = x.in_range(x1, v_thr)
            if ir != 0:
                # enter here if current interval is steep enough to generate spike(s)
                # 1- Get time of first spike (t0)
                t0 = x.g(x1 + ir * v_thr)

                # 2- Calculate how many spikes occur (k+1) and last spike (te)
                k = np.floor_divide((i + 1) * dt - t0, v_thr / abs(x.m) + t_ref)
                te = t0 + k * (t_ref + v_thr / abs(x.m))

                # 3- Update 'y' based on number of spikes in this interval
                y[i + 1] = y[i] + (k + 1) * np.sign(x.m) * v_thr

                spike_time = [t0]  # first spike
                isi = v_thr / abs(x.m)
                spike_time.extend(t0 + np.arange(1, k + 1) * (isi + t_ref))
                # 4- Update time from which err signal starts to count
                t1 = te + t_ref
                # 5- Get val of signal when err starts to count, or leave for future
                x1 = x.f(t1) if t1 < (i + 1) * dt else None

                # DEBUG: check on spike_time
                spike_time_isi = np.diff(spike_time)
                if np.any(spike_time_isi < t_ref):
                    print("PROBLEM with spike_time\n")

                # 6- Record the number of spikes in this sampling period + add spike times to list
                if np.sign(x.m) > 0:
                    spike_up.append(k + 1)
                    spike_dn.append(0)
                    spike_time_up.extend(spike_time)

                else:

                    spike_up.append(0)
                    spike_dn.append(k + 1)
                    spike_time_dn.extend(spike_time)

            else:
                # slope is not steep enough in current time interval
                y[i + 1] = y[i]
                # therefore no spikes
                spike_up.append(0)
                spike_dn.append(0)
        else:
            # current time interval occurs before refractory period has passed
            y[i + 1] = y[i]
            # therefore no spikes
            spike_up.append(0)
            spike_dn.append(0)
    check_sp_ref_period(spike_up, spike_dn, t_ref)

    return y, spike_up, spike_dn, spike_time_up, spike_time_dn


def spike_conversion(X, y, V_THR, T_REF, dt):
    n_samples, n_chs = X.shape[0], X.shape[1]
    samples_in_trial = int(MOV_DUR_IN_SEC * SAMP_FREQ)
    n_trials = int(n_samples / samples_in_trial)

    print("Spike conversion: window_len:{}".format(samples_in_trial))
    print("total dataset samples:{}  n_trials:{}\n\n".format(n_samples, n_trials))

    reconst = np.zeros(X.shape)  # initialize Spikes to store the reconstructed data matrix

    total_spikes = 0  # count number of spikes in the dataset
    trials_sp_count = np.zeros((n_trials, n_chs, 2))  # col 0: up spikes   col 1: dn spikes
    list_ch_spikes_model = []
    for ch in range(n_chs):
        ch_spike_count = 0  # count number of spikes in this channel
        trials_label = []
        list_trials_spikes_model = []
        for trial, i in enumerate(range(0, n_samples, samples_in_trial)):
            # print("trial:{}   i:{} nsamples:{} samples_in_trial:{} \n--------\n ".format(trial,i,n_samples,
            # samples_in_trial))
            # convert the signal in the current trial into spikes
            trial_data = X[i:(i + samples_in_trial), ch]
            trial_reconst, spike_up, spike_dn, spike_time_up, spike_time_dn = digitalize_sigma_J(trial_data, V_THR,
                                                                                                 T_REF, dt)
            # digitalize_sigma_k(trial_data, time, thr_up, thr_dn, t_ref)

            # t_up, t_dn = digitalize_sigma_new(trial_data, V_THR, V_THR, RESAMPLE_FREQUENCY, T_REF)

            # Using numba
            t_up, t_dn, times_interpolated, spike_idx_up, spike_idx_dn = ADM(trial_data, V_THR, V_THR, SAMP_FREQ, T_REF,
                                                                             return_indices=True, index_dt=1e-4)

            # Debug isi up/ dn
            isi_up = np.diff(t_up)
            isi_dn = np.diff(t_dn)
            if np.any(isi_up < T_REF):
                print("UP ch:{} and trial:{} isi > TREF".format(ch, trial))
                min_isi_loc = np.argwhere(isi_up < T_REF).flatten()
                first_sp = np.asarray(t_up)[min_isi_loc]
                second_sp = np.asarray(t_up)[min_isi_loc + 1]
                print("ISI:\n{}\n".format(second_sp - first_sp))
            if np.any(isi_dn < T_REF):
                print("DN ch:{} and trial:{} isi > TREF".format(ch, trial))
                min_isi_loc = np.argwhere(isi_dn < T_REF).flatten()
                first_sp = np.asarray(t_dn)[min_isi_loc]
                second_sp = np.asarray(t_dn)[min_isi_loc + 1]
                print("ISI:\n{}\n".format(second_sp - first_sp))

            reconst[i:(i + samples_in_trial), ch] = trial_reconst

            # list_trials_spikes_model.append(TrialSpikesModel(spike_time_up, spike_time_dn))
            list_trials_spikes_model.append(TrialSpikesModel(t_up, t_dn))

            ch_spike_count += np.sum(spike_up) + np.sum(spike_dn)  # count spikes in this trial: up + dn
            trials_sp_count[trial, ch, :] = [np.sum(spike_up), np.sum(spike_dn)]
            trials_label.append(y[i])

        list_ch_spikes_model.append(ChannelSpikesModel(list_trials_spikes_model))
        total_spikes += ch_spike_count  # add ch spike count to the total spikes in the dataset

    # calculate the spiking frequency
    spiking_frequency = int((total_spikes / n_chs) / (n_samples / SAMP_FREQ))
    assert reconst.shape == X.shape
    return reconst, spiking_frequency, trials_sp_count, trials_label, list_ch_spikes_model
