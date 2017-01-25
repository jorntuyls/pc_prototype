from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


from pc_smac.pc_smac.pipeline_space.node import Node

from pc_smac.pc_smac.utils.constants import *

class ImputationNode(Node):

    def __init__(self):
        self.name = "imputation"
        self.type = "data_preprocessor"
        self.hyperparameters = {"strategy": "mean"}

    def initialize_algorithm(self, hyperparameters):
        from sklearn.preprocessing import Imputer
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        imputation = Imputer(strategy=hyperparameters["strategy"], copy=False)
        return (self.get_full_name(), imputation)

    def get_hyperparameter_search_space(self):
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter(
            "strategy", ["mean", "median", "most_frequent"], default="mean")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)
        return cs

    def get_properties(self):
        return {'shortname': 'Imputation',
                'name': 'Imputation',
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