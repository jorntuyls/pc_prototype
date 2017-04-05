
import abc

from smac.tae.execute_ta_run import StatusType
from smac.tae.execute_ta_run import ExecuteTARun

class WhiteBoxFunction(ExecuteTARun):

    __metaclass__ = abc.ABCMeta

    def __init__(self, runhistory, statistics):
        self.runhistory = runhistory
        self.statistics = statistics
        self.smac_stats = None

    def set_smac_stats(self, smac_stats):
        self.smac_stats = smac_stats

    def start(self,
              config,
              instance,
              cutoff=None,
              seed=12345,
              instance_specific="0",
              capped=False):
        x = config["preprocessor:x"]  # preprocessor
        y = config["classifier:y"]  # classifier

        # Calculate cost and set status
        cost = self.z_func(x, y)
        status = StatusType.SUCCESS

        # Calculate runtime
        runtime = x + y
        # Hack stats
        self.statistics.hack_time(runtime)
        if self.smac_stats:
            self.smac_stats.hack_time(runtime)

        # Add information of this run to statistics
        run_information = {
            'cost': cost,
            'runtime': runtime,
            'pipeline_steps_timing': {'preprocessor:x': x, 'classifier:y': y},
        }
        self.statistics.add_run(config.get_dictionary(), run_information, config_origin=config.origin)

        # Additional info
        additional_info = {}

        if self.runhistory:
            self.runhistory.add(config=config,
                                cost=cost,
                                time=runtime,
                                status=status,
                                instance_id=instance,
                                seed=seed,
                                additional_info=additional_info)

        return status, cost, runtime, additional_info

    @abc.abstractmethod
    def z_func(self, x, y):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_config_space(self, seed=None):
        raise NotImplementedError()

class CachedWhiteboxFunction(WhiteBoxFunction):

    def __init__(self, runhistory, statistics):
        self.cache = set()
        self.cache_hits = 0
        self.total_evaluations = 0
        super(CachedWhiteboxFunction, self).__init__(runhistory, statistics)

    def start(self,
              config,
              instance,
              cutoff=None,
              seed=12345,
              instance_specific="0",
              capped=False):
        # Increase the number of total evaluations
        self.total_evaluations += 1

        x = config["preprocessor:x"]  # preprocessor
        y = config["classifier:y"]  # classifier

        # Calculate cost and set status
        cost = self.z_func(x, y)
        status = StatusType.SUCCESS

        # Calculate runtime
        if x in self.cache:
            runtime = y
            self.cache_hits += 1
        else:
            runtime = x + y
        # Hack stats
        self.statistics.hack_time(runtime)
        if self.smac_stats:
            self.smac_stats.hack_time(runtime)

        # Add preprocessor to cache
        self.cache.update({x})

        # Add information of this run to statistics
        run_information = {
            'cost': cost,
            'runtime': runtime,
            'pipeline_steps_timing': {'preprocessor:x': x, 'classifier:y': y},
            'cache_hits': self.cache_hits,
            'total_evaluations': self.total_evaluations
        }
        self.statistics.add_run(config.get_dictionary(), run_information, config_origin=config.origin)

        # Setup additional info
        # t_rc is a list of tuples (dict, time)
        t_rc = [({'preprocessor:x': x}, x)]
        additional_info = {'t_rc': t_rc}

        if self.runhistory:
            self.runhistory.add(config=config,
                                cost=cost,
                                time=runtime,
                                status=status,
                                instance_id=instance,
                                seed=seed,
                                additional_info=additional_info)

        return status, cost, runtime, additional_info
