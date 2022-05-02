class ChannelSpikesModel:
    """Class organizing the ADM UP and DOWN (DN) spikes (times and neuron ids) from each TIME channel."""

    def __init__(self, trials_model, subject):
        """
        :param list[TrialSpikesModel] trials_model: list of TrialSpikesModel object modelling the trials for the TIME
        channel
        """
        self.list_trials_model = trials_model
        self.trials_count = len(trials_model)
        self.subject = subject

    def get_spikes_for_trial(self, trial_id):
        """
        Retrieves thea UP and DN spike streams for the given trial_id.
        :param int trial_id: trial order in the
        experiment (i.e global trial order and not relative to the gesture trials)
        :rtype: (list[float], list[float])
        :return: ordered UP spike times for this trial, ordered DN spike times for this trial
        """
        return self.list_trials_model[trial_id].sp_up, self.list_trials_model[trial_id].sp_dn

    def get_spikes_count_for_trial(self, trial_id):
        """
        Retrieves the number of spikes (events) for the given trial_id.
        :param int trial_id: trial order in the experiment (i.e global trial order and not relative to the gesture
        trials)
        :return: (int,int) UP spike count, DN spike count
        """
        up_sp, dn_sp = self.get_spikes_for_trial(trial_id)
        return len(up_sp), len(dn_sp)


class TrialSpikesModel:
    """Class organizing the ADM UP and DOWN (DN) spike outputs (times and neuron ids) for each gesture trial."""

    def __init__(self, sp_up, sp_dn):
        """
        Sets the UP and DN spike times for the trial.
        :param list[float] sp_up: ordered UP spike times for the trial
        :param list[float] sp_dn: ordered DN spike times for the trial
        """
        self.sp_up = sp_up
        self.sp_dn = sp_dn
