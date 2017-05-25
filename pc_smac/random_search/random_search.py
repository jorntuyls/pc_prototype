
import inspect
import logging
import pynisher

import random
import numpy as np

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
        _, incumbent_cost, _, _ = self.run_with_limits(incumbent, instance=1, cutoff=cutoff, seed=None)
        self.statistics.add_run_nb()
        self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        while not(self.statistics.is_budget_exhausted()):
            rand_config = self.config_space.sample_configuration()
            _, cost, _, additional_info = self.run_with_limits(rand_config, instance=1, cutoff=cutoff, seed=None)
            self.statistics.add_run_nb()

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
                 splitting_number,
                 random_splitting_enabled=False):
        self.splitting_number = splitting_number
        self.random_splitting_enabled = random_splitting_enabled
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
        _, incumbent_cost, _, _ = self.run_with_limits(incumbent, instance=1, cutoff=cutoff, seed=None)
        self.statistics.add_run_nb()
        self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        while not(self.statistics.is_budget_exhausted()):
            configs = self.sample_batch_of_configurations()

            for config in configs:
                _, cost, additional_info, _ = self.run_with_limits(config, instance=1, cutoff=cutoff, seed=None)
                self.statistics.add_run_nb()

                if cost < incumbent_cost:
                    incumbent = config
                    incumbent_cost = cost
                    self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})
            test = False

        return incumbent

    def sample_batch_of_configurations(self):
        start_config = self.config_space.sample_configuration()
        batch_of_configs = [start_config]

        if self.random_splitting_enabled:
            batch_size = np.random.randint(1, self.splitting_number)
        else:
            batch_size = self.splitting_number

        while len(batch_of_configs) < batch_size:
            next_config = self.config_space.sample_configuration()
            complemented_vector_values = self._get_vector_values(next_config, self.variable_pipeline_steps)
            # TODO hack for now to combine preprocessing part of one configuration with classification part of all the others
            next_config_combined = self._combine_configurations_batch_vector(start_config, [complemented_vector_values])
            batch_of_configs.extend(next_config_combined)
        return batch_of_configs

    def _combine_configurations_batch_vector(self, start_config, complemented_configs_values):
        constant_vector_values = self._get_vector_values(start_config, self.constant_pipeline_steps)
        batch = []

        for complemented_config_values in complemented_configs_values:
            vector = np.ndarray(len(self.config_space._hyperparameters),
                                dtype=np.float)

            vector[:] = np.NaN

            for key in constant_vector_values:
                vector[key] = constant_vector_values[key]

            for key in complemented_config_values:
                vector[key] = complemented_config_values[key]

            try:
                self.config_space._check_forbidden(vector)
                config_object = Configuration(configuration_space=self.config_space,
                                              vector=vector)
                batch.append(config_object)
            except ValueError as v:
                pass

        return batch

    def _get_values(self, config_dict, pipeline_steps):
        value_dict = {}
        for step_name in pipeline_steps:
            for hp_name in config_dict:
                splt_hp_name = hp_name.split(":")
                if splt_hp_name[0] == step_name:
                    value_dict[hp_name] = config_dict[hp_name]
        return value_dict

    def _get_vector_values(self, config, pipeline_steps):
        vector = config.get_array()
        value_dict = {}
        for hp_name in config.get_dictionary():
            for step_name in pipeline_steps:
                splt_hp_name = hp_name.split(":")
                if splt_hp_name[0] == step_name:
                    item_idx = self.config_space._hyperparameter_idx[hp_name]
                    value_dict[item_idx] = vector[item_idx]
        return value_dict


class SigmoidRandomSearch(RandomSearch):

    def __init__(self,
                 config_space,
                 pipeline_runner,
                 wallclock_limit,
                 memory_limit,
                 statistics,
                 constant_pipeline_steps,
                 variable_pipeline_steps,
                 splitting_number,
                 random_splitting_enabled=False):
        self.splitting_number = float(splitting_number)
        self.random_splitting_enabled = random_splitting_enabled
        self.constant_pipeline_steps = constant_pipeline_steps
        self.variable_pipeline_steps = variable_pipeline_steps
        super(SigmoidRandomSearch, self).__init__(config_space=config_space,
                                               pipeline_runner=pipeline_runner,
                                               wallclock_limit=wallclock_limit,
                                               memory_limit=memory_limit,
                                               statistics=statistics)

    def run(self, cutoff):

        start_time = self.statistics.start_timer()

        incumbent = self.config_space.sample_configuration()
        _, incumbent_cost, _, _ = self.run_with_limits(incumbent, instance=1, cutoff=cutoff, seed=None)
        self.statistics.add_run_nb()
        self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        incumbent_lst = []
        random_timing = 0
        incumbent_timing = 0
        while not(self.statistics.is_budget_exhausted()):
            sigmoid_value = 1.0/(1.0 + np.exp(-len(incumbent_lst)/self.splitting_number+2))
            if incumbent_lst != [] and incumbent_timing < sigmoid_value * (incumbent_timing + random_timing):
                start_config = random.choice(incumbent_lst)
                challenger_lst = []
                while challenger_lst == []:
                    rand_config = self.config_space.sample_configuration()
                    complemented_vector_values = self._get_vector_values(rand_config, self.variable_pipeline_steps)
                    challenger_lst = self._combine_configurations_batch_vector(start_config, [complemented_vector_values])
                challenger = challenger_lst[0]
                _, cost, runtime, _ = self.run_with_limits(challenger, instance=1, cutoff=cutoff, seed=None)
                incumbent_timing += runtime
            else:
                challenger = self.config_space.sample_configuration()
                _, cost, runtime, _ = self.run_with_limits(challenger, instance=1, cutoff=cutoff, seed=None)
                random_timing += runtime

            self.statistics.add_run_nb()

            if cost < incumbent_cost:
                random_timing = 0
                incumbent_timing = 0
                incumbent = challenger
                incumbent_cost = cost
                #for _ in range(len(incumbent_lst) + 1):
                incumbent_lst.append(incumbent)
                self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        return incumbent

    def _combine_configurations_batch_vector(self, start_config, complemented_configs_values):
        constant_vector_values = self._get_vector_values(start_config, self.constant_pipeline_steps)
        batch = []

        for complemented_config_values in complemented_configs_values:
            vector = np.ndarray(len(self.config_space._hyperparameters),
                                dtype=np.float)

            vector[:] = np.NaN

            for key in constant_vector_values:
                vector[key] = constant_vector_values[key]

            for key in complemented_config_values:
                vector[key] = complemented_config_values[key]

            try:
                self.config_space._check_forbidden(vector)
                config_object = Configuration(configuration_space=self.config_space,
                                              vector=vector)
                batch.append(config_object)
            except ValueError as v:
                pass

        return batch

    def _get_vector_values(self, config, pipeline_steps):
        vector = config.get_array()
        value_dict = {}
        for hp_name in config.get_dictionary():
            for step_name in pipeline_steps:
                splt_hp_name = hp_name.split(":")
                if splt_hp_name[0] == step_name:
                    item_idx = self.config_space._hyperparameter_idx[hp_name]
                    value_dict[item_idx] = vector[item_idx]
        return value_dict