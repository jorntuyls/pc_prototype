
from ConfigSpace.configuration_space import ConfigurationSpace

from scipy import sparse

from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingNode
from pc_smac.pc_smac.utils.constants import *

class StandardScalerNode(Node):

    def __init__(self):
        self.name = "standard_scaler"
        self.type = "data_preprocessor"
        self.hyperparameters = {}
        self.algorithm = StandardScaler

    def initialize_algorithm(self, hyperparameters):
        standard_scaler = self.algorithm(random_state=None)
        return (self.get_full_name(), standard_scaler)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class Rescaling(object):
    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


class StandardScaler(Rescaling, PreprocessingNode):
    def __init__(self, random_state):
        from sklearn.preprocessing import StandardScaler
        self.preprocessor = StandardScaler()

    def fit(self, X, y=None):
        if sparse.isspmatrix(X):
            self.preprocessor.set_params(with_mean=False)

        return super(StandardScaler, self).fit(X, y)

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



