
from smac.runhistory.runhistory import RunHistory

class PCRunHistory(RunHistory):

    def __init__(self, aggregate_func):
        self.cached_configurations = []
        super(PCRunHistory, self).__init__(aggregate_func)

    def add(self, config, cost, time,
            status, instance_id=None,
            seed=None,
            additional_info=None):

        #print("BEGIN PCRUNHISTORY")

        #print(self.cached_configurations)

        if additional_info and 't_rc' in additional_info.keys():
            # additional_info['t_rc'] is a list of tuples (dict, time) where dict is a cached algorithm (part of pipeline)
            #   configuration and time is runtime that this algorithm configuration took
            for cached_config, runtime in additional_info['t_rc']:
                if not self._contains_config(self.get_cached_configurations_list(), cached_config):
                    # if the configurations are already in the cached configurations, then we don't want to add them again
                    self.cached_configurations.append((cached_config, runtime))
                #else:
                    #print("deleted config: {}, {}".format(cached_config, runtime))

        #print("BETWEEN")
        #print("config cost: {}, {}".format(config, time))
        #print(self.cached_configurations)

        # If a configuration is added that exists in the cached configurations, then we want to add the dicounted cost again for the model
        #   this ensures that the model is consistent
        for cached_config, runtime_discount in self.get_cached_configurations():
            if self._partially_contains_config(config, cached_config):
                time += runtime_discount

        super(PCRunHistory, self).add(config, cost, time, status, instance_id, seed, additional_info)
        #print("config cost: {}, {}".format(config, time))
        #print("END PCRUNHISTORY")

    def get_cached_configurations(self):
        return self.cached_configurations

    def get_cached_configurations_list(self):
        return [key[0] for key in self.get_cached_configurations()]

    def get_all_configs(self):
        return list(self.config_ids.keys())

    #### PRIVATE METHODS ####

    def _partially_contains_config(self, config1, config2):
        '''

        Parameters
        ----------
        config1
        config2

        Returns
        -------
            if config1 contains config2, but config1 can also contains more hyperparameters

        '''
        r = [key for key in config2.keys() if config2[key] != config1[key]]
        if r == []:
            return True
        return False

    def _contains_config(self, config_list, config):
        return_value = False
        for config_l in config_list:
            if self._is_equal(config_l, config):
                return_value = True
        return return_value

    def _is_equal(self, config1, config2):
        r = [key for key in config1.keys() if (key not in config2.keys() or config1[key] != config2[key])]
        s = [key for key in config2.keys() if (key not in config1.keys() or config2[key] != config1[key])]
        if s == [] and r == []:
            return True
        return False