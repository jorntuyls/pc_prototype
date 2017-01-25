
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

from pc_smac.pc_smac.pipeline_space.node import Node

from pc_smac.pc_smac.utils.constants import *

class OneHotEncodeingNode(Node):

    def __init__(self):
        self.name = "one_hot_encoding"
        self.type = "data_preprocessor"
        self.hyperparameters = {"use_minimum_fraction": "True", "minimum_fraction": 0.01}

    def initialize_algorithm(self, hyperparameters):
        from pc_smac.pc_smac.algorithms.data_preprocessing.one_hot_encoding import OneHotEncoder
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        one_hot_encoder = OneHotEncoder(use_minimum_fraction=bool(hyperparameters["use_minimum_fraction"]),
                                        minimum_fraction=float(hyperparameters["minimum_fraction"]))

        return (self.get_full_name(), one_hot_encoder)

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()
        use_minimum_fraction = cs.add_hyperparameter(CategoricalHyperparameter(
            "use_minimum_fraction", ["True", "False"], default="True"))
        minimum_fraction = cs.add_hyperparameter(UniformFloatHyperparameter(
            "minimum_fraction", lower=.0001, upper=0.5, default=0.01, log=True))
        cs.add_condition(EqualsCondition(minimum_fraction,
                                         use_minimum_fraction, 'True'))
        return cs

    def get_properties(self):
        return {'shortname': '1Hot',
                'name': 'One Hot Encoder',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}