class TrialSpikesModel:
    """Class organizing the ADM UP and DOWN (DN) spike outputs (times and neuron ids) for each trial/repetition."""

    def __init__(self, list_ch_model):
        self.list_ch_model = list_ch_model
        self.n_ch = len(list_ch_model)

        self.trial_times, self.trials_ids = self._get_spikes_for_trial()

        # Todo: get the unmerged spike times up and dn separetely to facilitate analysis on enc

    def _get_spikes_for_trial(self):
        sp_up = []
        sp_dn = []
        sp_id_up = []
        sp_id_dn = []

        for ch in range(self.n_ch):
            sp_up.append(self.list_ch_model[ch].sp_up)
            sp_dn.append(self.list_ch_model[ch].sp_dn)

            sp_id_up.append(self.list_ch_model[ch].sp_id_up)
            sp_id_dn.append(self.list_ch_model[ch].sp_id_dn)

        # merge and sort
        unsorted_sp_times = sp_up + sp_dn
        unsorted_sp_indices = sp_id_up + sp_id_dn

        # sort spike times
        trial_times, trial_ids = map(list, zip(*sorted(zip(unsorted_sp_times, unsorted_sp_indices), reverse=False)))
        return trial_times, trial_ids


class ChannelSpikesModel:
    """Class organizing the ADM UP and DOWN (DN) spikes (times and neuron ids) from each EMG channel."""

    def __init__(self, sp_time_up, sp_id_up, sp_time_dn, sp_id_dn):
        self.sp_up = sp_time_up
        self.sp_id_up = sp_id_up

        self.sp_dn = sp_time_dn
        self.sp_id_dn = sp_id_dn
