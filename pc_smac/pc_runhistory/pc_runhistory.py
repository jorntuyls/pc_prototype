
from smac.runhistory.runhistory import RunHistory

class PCRunHistory(RunHistory):

    def __init__(self, aggregate_func):
        self.cached_configurations = []
        super(PCRunHistory, self).__init__(aggregate_func)

    def add(self, config, cost, time,
            status, instance_id=None,
            seed=None,
            additional_info=None):
        super(PCRunHistory, self).add(config, cost, time, status, instance_id, seed, additional_info)

        if additional_info and 't_rc' in additional_info.keys():
            self.cached_configurations += additional_info['t_rc']

    def get_cached_configurations(self):
        return self.cached_configurations

    def get_configs_from_previous_runs(self):
        return list(self.config_ids.keys())