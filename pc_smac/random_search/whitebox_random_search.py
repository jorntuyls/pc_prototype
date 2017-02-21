
import time

from ConfigSpace.configuration_space import Configuration

# TODO Combine this with random search for pipelines
class WhiteboxRandomSearch(object):

    def __init__(self,
                 config_space,
                 pipeline_runner,
                 wallclock_limit,
                 statistics):
        self.config_space = config_space
        self.pipeline_runner = pipeline_runner
        self.wallclock_limit = wallclock_limit
        self.statistics = statistics

    def run(self):

        start_time = self.statistics.start_timer()

        incumbent = self.config_space.sample_configuration()
        _, incumbent_cost, _, _ = self.pipeline_runner.start(incumbent, instance=None, seed=None)
        self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        while not(self.statistics.is_budget_exhausted()):
            rand_config = self.config_space.sample_configuration()
            _, cost, _, _ = self.pipeline_runner.start(rand_config, instance=None, seed=None)
            if cost < incumbent_cost:
                incumbent = rand_config
                incumbent_cost = cost
                self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        return incumbent

class Whitebox2stepRandomSearch(WhiteboxRandomSearch):

    def __init__(self,
                 config_space,
                 pipeline_runner,
                 wallclock_limit,
                 statistics,
                 constant_pipeline_steps,
                 variable_pipeline_steps,
                 number_leafs_split):
        self.number_leafs_split = number_leafs_split
        self.constant_pipeline_steps = constant_pipeline_steps
        self.variable_pipeline_steps = variable_pipeline_steps
        super(Whitebox2stepRandomSearch, self).__init__(config_space=config_space,
                                               pipeline_runner=pipeline_runner,
                                               wallclock_limit=wallclock_limit,
                                               statistics=statistics)

    def run(self):

        start_time = self.statistics.start_timer()

        incumbent = self.config_space.sample_configuration()
        _, incumbent_cost, _, _ = self.pipeline_runner.start(incumbent, instance=None, seed=None)
        self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

        while not(self.statistics.is_budget_exhausted()):
            rand_configs = self.config_space.sample_configuration(size=self.number_leafs_split)
            # TODO hack for now to combine preprocessing part of one configuration with classification part of all the others
            new_configs = self.combine_configurations(rand_configs)

            for config in new_configs:
                _, cost, _, _ = self.pipeline_runner.start(config, instance=None, seed=None)
                if cost < incumbent_cost:
                    incumbent = config
                    incumbent_cost = cost
                    self.statistics.add_new_incumbent(incumbent.get_dictionary(), {'cost': incumbent_cost})

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

