
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


from pc_smac.pc_smac.pipeline_space.node import Node

from pc_smac.pc_smac.utils.constants import *


class Balancing(Node):

    def __init__(self):
        self.name = "balancing"
        self.type = "data_preprocessor"
        self.hyperparameters = {"strategy": "none"}

    def initialize_algorithm(self, hyperparameters):
        from pc_smac.pc_smac.algorithms.data_preprocessing.balancing_algo import Balancing
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        balancing = Balancing(strategy=hyperparameters["strategy"])
        return (self.get_full_name(), balancing)

    def get_hyperparameter_search_space(self):
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter(
            "strategy", ["none", "weighting"], default="none")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)
        return cs

    def get_properties(self):
        return {'shortname': 'Balancing',
                'name': 'Balancing Imbalanced Class Distributions',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

