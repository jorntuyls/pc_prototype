from ConfigSpace.configuration_space import ConfigurationSpace
from scipy import sparse

from pc_smac.pc_smac.pipeline_space.data_preprocessing_nodes.rescaling.abstract_rescaling import Rescaling
from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *


class StandardScalerNode(Node):

    def __init__(self):
        self.name = "standard_scaler"
        self.type = "rescaling"
        self.hyperparameters = {}
        self.algorithm = StandardScaler

    def initialize_algorithm(self, hyperparameters):
        standard_scaler = self.algorithm(random_state=None)
        return (self.get_full_name(), standard_scaler)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class StandardScaler(Rescaling, PreprocessingAlgorithm):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            'random_state': self.random_state
        }

    def fit(self, X, y=None):
        from sklearn.preprocessing import StandardScaler
        self.preprocessor = StandardScaler()
        if sparse.isspmatrix(X):
            self.preprocessor.set_params(with_mean=False)

        return self.preprocessor.fit(X, y)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'StandardScaler',
                'name': 'StandardScaler',
                'handles_missing_values': False,
                'handles_nominal_values': False,
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
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs



