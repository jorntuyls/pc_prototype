from pc_smac.pc_smac.pipeline_space.data_preprocessing_nodes.rescaling.abstract_rescaling import Rescaling
from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *


class NoRescalingNode(Node):

    def __init__(self):
        self.name = "no_rescaling"
        self.type = "rescaling"
        self.hyperparameters = {}
        self.algorithm = NoRescaling

    def initialize_algorithm(self, hyperparameters):
        no_rescaler = self.algorithm(random_state=None)
        return (self.get_full_name(), no_rescaler)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class NoRescaling(Rescaling, PreprocessingAlgorithm):
    def __init__(self, random_state):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MinMaxScaler',
                'name': 'MinMaxScaler',
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