
from ConfigSpace.configuration_space import ConfigurationSpace

from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingNode
from pc_smac.pc_smac.utils.constants import *

class NoPreprocessingNode(Node):

    def __init__(self):
        self.name = "no_preprocessing"
        self.type = "feature_preprocessor"
        self.hyperparameters = {}
        self.algorithm = NoPreprocessing

    def initialize_algorithm(self, hyperparameters):
        return (self.get_full_name(), self.algorithm(None))

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class NoPreprocessing(PreprocessingNode):

    def __init__(self, random_state):
        """ This preprocessors does not change the data """
        self.preprocessor = None

    def fit(self, X, Y=None):
        self.preprocessor = 0
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'no',
                'name': 'NoPreprocessing',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


