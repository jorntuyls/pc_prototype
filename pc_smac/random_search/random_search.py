
import inspect
import logging
import pynisher

from ConfigSpace.configuration_space import Configuration

from smac.tae.execute_ta_run import StatusType

class RandomSearch(object):

    def __init__(self,
                 config_space,
                 pipeline_runner,
                 wallclock_limit,
                 memory_limit,
                 statistics):
        self.config_space = config_space
        self.pipeline_runner = pipeline_runner
        self.wallclock_limit = wallclock_limit
        self.memory_limit = memory_limit
        self.statistics = statistics

        self.ta = self.pipeline_runner.run

        signature = inspect.signature(self.ta).parameters
        self._accepts_seed = len(signature) > 1
        self._accepts_instance = len(signature) > 2

    def run(self, cutoff):

        start_time = self.statistics.start_timer()

        incumbent = self.config_space.sample_configuration()
        _, incumbent_cost, _, _ = self.run_with_limits(incumbent, instance=None, cutoff=cutoff, seed=None)
        self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        while not(self.statistics.is_budget_exhausted()):
            rand_config = self.config_space.sample_configuration()
            _, cost, _, additional_info = self.run_with_limits(rand_config, instance=None, cutoff=cutoff, seed=None)
            if cost < incumbent_cost:
                incumbent = rand_config
                incumbent_cost = cost
                self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        return incumbent

    def run_with_limits(self,
                        config,
                        instance=None,
                        cutoff=None,
                        seed=12345):
        arguments = {'logger': logging.getLogger("pynisher"),
                     'wall_time_in_s': cutoff,
                     'mem_in_mb': self.memory_limit}

        obj = pynisher.enforce_limits(**arguments)(self.ta)

        obj_kwargs = {}
        if self._accepts_seed:
            obj_kwargs['seed'] = seed
        if self._accepts_instance:
            obj_kwargs['instance'] = instance

        rval = obj(config, **obj_kwargs)

        if isinstance(rval, tuple):
            result = rval[0]
            additional_run_info = rval[1]
        else:
            result = rval
            additional_run_info = {}

        if obj.exit_status is pynisher.TimeoutException:
            status = StatusType.TIMEOUT
            cost = 1234567890
        elif obj.exit_status is pynisher.MemorylimitException:
            status = StatusType.MEMOUT
            cost = 1234567890
        elif obj.exit_status == 0 and result is not None:
            status = StatusType.SUCCESS
            cost = result
        else:
            status = StatusType.CRASHED
            cost = 1234567890  # won't be used for the model

        runtime = float(obj.wall_clock_time)

        return status, cost, runtime, additional_run_info

class TreeRandomSearch(RandomSearch):

    def __init__(self,
                 config_space,
                 pipeline_runner,
                 wallclock_limit,
                 memory_limit,
                 statistics,
                 constant_pipeline_steps,
                 variable_pipeline_steps,
                 number_leafs_split):
        self.number_leafs_split = number_leafs_split
        self.constant_pipeline_steps = constant_pipeline_steps
        self.variable_pipeline_steps = variable_pipeline_steps
        super(TreeRandomSearch, self).__init__(config_space=config_space,
                                               pipeline_runner=pipeline_runner,
                                               wallclock_limit=wallclock_limit,
                                               memory_limit=memory_limit,
                                               statistics=statistics)

    def run(self, cutoff):

        start_time = self.statistics.start_timer()

        incumbent = self.config_space.sample_configuration()
        _, incumbent_cost, _, _ = self.run_with_limits(incumbent, instance=None, cutoff=cutoff, seed=None)
        self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        while not(self.statistics.is_budget_exhausted()):
            rand_configs = self.config_space.sample_configuration(size=self.number_leafs_split)
            # TODO hack for now to combine preprocessing part of one configuration with classification part of all the others
            new_configs = self.combine_configurations(rand_configs)

            for config in new_configs:
                _, cost, additional_info, _ = self.run_with_limits(config, instance=None, cutoff=cutoff, seed=None)
                if cost < incumbent_cost:
                    incumbent = config
                    incumbent_cost = cost
                    self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})
            test = False

        return incumbent


    def combine_configurations(self, configs):
        default = configs.pop()
        constant_values = self._get_values(default.get_dictionary(), self.constant_pipeline_steps)
        new_configs = [default]

        for config in configs:
            new_config_values = {}
            new_config_values.update(constant_values)
            variable_values = self._get_values(config.get_dictionary(), self.variable_pipeline_steps)
            new_config_values.update(variable_values)
            new_configs.append(Configuration(configuration_space=self.config_space,
                                             values=new_config_values))

        return new_configs


    def _get_values(self, config_dict, pipeline_steps):
        value_dict = {}
        for step_name in pipeline_steps:
            for hp_name in config_dict:
                splt_hp_name = hp_name.split(":")
                if splt_hp_name[0] == step_name:
                    value_dict[hp_name] = config_dict[hp_name]
        return value_dict

