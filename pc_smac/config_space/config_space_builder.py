
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class ConfigSpaceBuilder:

    def __init__(self, pipeline_space):
        self.pipeline_space = pipeline_space

    def build_config_space(self, seed=None):
        cs = ConfigurationSpace() if seed == None else ConfigurationSpace(seed=seed)
        for ps in self.pipeline_space.get_pipeline_steps():
            cs.add_hyperparameter(CategoricalHyperparameter(ps.get_name(),
                ps.get_node_names()))
            for node in ps.get_nodes():
                sub_cs = node.get_hyperparameter_search_space()
                cs.add_configuration_space(node.get_name(), sub_cs)

        print(cs)
        return cs

