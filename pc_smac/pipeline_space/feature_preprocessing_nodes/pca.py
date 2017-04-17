
# The hyperparameter spaces come from https://github.com/automl/auto-sklearn

import warnings
import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *

class PcaNode(Node):

    def __init__(self):
        self.name = "pca"
        self.type = "feature_preprocessor"
        self.hyperparameters = {"keep_variance": 0.9999, "whiten": "False"}
        self.algorithm = PCA

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        pca = self.algorithm(keep_variance=hyperparameters["keep_variance"],
                            whiten=hyperparameters["whiten"])

        return (self.get_full_name(), pca)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class PCA(PreprocessingAlgorithm):
    def __init__(self, keep_variance, whiten, random_state=None):
        self.keep_variance = keep_variance
        self.whiten = whiten
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            'keep_variance': self.keep_variance,
            'whiten': self.whiten,
            'random_state': self.random_state
        }

    def fit(self, X, Y=None):
        import sklearn.decomposition
        n_components = float(self.keep_variance)
        self.preprocessor = sklearn.decomposition.PCA(n_components=n_components,
                                                      whiten=self.whiten,
                                                      copy=True)
        self.preprocessor.fit(X)

        if not np.isfinite(self.preprocessor.components_).all():
            raise ValueError("PCA found non-finite components.")

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PCA',
                'name': 'Principle Component Analysis',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO document that we have to be very careful
                'is_deterministic': False,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        keep_variance = UniformFloatHyperparameter(
            "keep_variance", 0.5, 0.9999, default=0.9999)
        whiten = CategoricalHyperparameter(
            "whiten", ["False", "True"], default="False")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(keep_variance)
        cs.add_hyperparameter(whiten)
        return cs

