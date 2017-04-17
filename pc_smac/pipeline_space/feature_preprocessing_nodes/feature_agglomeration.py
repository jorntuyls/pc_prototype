
import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from ConfigSpace.forbidden import ForbiddenInClause, \
    ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.conditions import EqualsCondition, InCondition

from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *

class FeatureAgglomerationNode(Node):
    def __init__(self):
        self.name = "feature_agglomeration"
        self.type = "feature_preprocessor"
        self.hyperparameters = {"n_clusters": 25, "affinity": "euclidean", "linkage": "ward",
                                "pooling_func": "mean"}

        self.algorithm = FeatureAgglomeration

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        feature_agglomeration = self.algorithm(n_clusters=hyperparameters["n_clusters"],
                                                 affinity=hyperparameters["affinity"],
                                                 linkage=hyperparameters["linkage"],
                                                 pooling_func=hyperparameters["pooling_func"])

        return (self.get_full_name(), feature_agglomeration)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class FeatureAgglomeration(PreprocessingAlgorithm):
    def __init__(self, n_clusters, affinity, linkage, pooling_func,
        random_state=None):
        self.n_clusters = int(n_clusters)
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func = pooling_func
        self.random_state = random_state

        self.pooling_func_mapping = dict(mean=np.mean,
                                         median=np.median,
                                         max=np.max)

    def get_params(self, deep=True):
        return {
            'n_clusters': self.n_clusters,
            'affinity': self.affinity,
            'linkage': self.linkage,
            'pooling_func': self.pooling_func,
            'random_state': self.random_state
        }

    def fit(self, X, Y=None):
        import sklearn.cluster

        n_clusters = min(self.n_clusters, X.shape[1])
        if not callable(self.pooling_func):
            self.pooling_func = self.pooling_func_mapping[self.pooling_func]

        self.preprocessor = sklearn.cluster.FeatureAgglomeration(
            n_clusters=n_clusters, affinity=self.affinity,
            linkage=self.linkage, pooling_func=self.pooling_func)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Feature Agglomeration',
                'name': 'Feature Agglomeration',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_clusters = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "n_clusters", 2, 400, 25))
        affinity = cs.add_hyperparameter(CategoricalHyperparameter(
            "affinity", ["euclidean", "manhattan", "cosine"], "euclidean"))
        linkage = cs.add_hyperparameter(CategoricalHyperparameter(
            "linkage", ["ward", "complete", "average"], "ward"))
        pooling_func = cs.add_hyperparameter(CategoricalHyperparameter(
            "pooling_func", ["mean", "median", "max"]))

        affinity_and_linkage = ForbiddenAndConjunction(
            ForbiddenInClause(affinity, ["manhattan", "cosine"]),
            ForbiddenEqualsClause(linkage, "ward"))
        cs.add_forbidden_clause(affinity_and_linkage)
        return cs




