
# The algorithm and hyperparameter spaces come from https://github.com/automl/auto-sklearn

import numpy as np
import sklearn.ensemble

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter

from pc_smac.pc_smac.pipeline_space.node import Node, ClassificationAlgorithm
from pc_smac.pc_smac.utils.constants import *

class GradientBoostingNode(Node):

    def __init__(self):
        self.name = "gradient_boosting"
        self.type = "classifier"
        self.hyperparameters = {"loss": "deviance", "learning_rate": 0.1, "n_estimators": 100, "max_depth": 3,
                                "min_samples_split": 2, "min_samples_leaf": 1, "min_weight_fraction_leaf": 0.0,
                                "subsample": 1.0, "max_features": 1, "max_leaf_nodes": "None"}
        self.algorithm = GradientBoosting

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        gradient_boosting = self.algorithm(loss=hyperparameters["loss"],
                                           learning_rate=hyperparameters["learning_rate"], # Actual hyperparameter
                                           n_estimators=hyperparameters["n_estimators"], # Actual hyperparameter
                                           subsample=hyperparameters["subsample"], # Actual hyperparameter
                                           min_samples_split=hyperparameters["min_samples_split"], # Actual hyperparameter
                                           min_samples_leaf=hyperparameters["min_samples_leaf"], # Actual hyperparameter
                                           min_weight_fraction_leaf=hyperparameters["min_weight_fraction_leaf"],
                                           max_depth=hyperparameters["max_depth"], # Actual hyperparameter
                                           max_features=hyperparameters["max_features"], # Actual hyperparameter
                                           max_leaf_nodes=hyperparameters["max_leaf_nodes"])

        return (self.get_full_name(), gradient_boosting)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)



class GradientBoosting(ClassificationAlgorithm):
    def __init__(self, loss, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_depth, max_features,
                 max_leaf_nodes, init=None, random_state=None, verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.fully_fit_ = False

    def fit(self, X, y, sample_weight=None, refit=False):
        if self.estimator is None or refit:
            self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight,
                               refit=refit)

        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight)
        return self

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):

        # Special fix for gradient boosting!
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=X.dtype)
        if refit:
            self.estimator = None

        if self.estimator is None:
            self.learning_rate = float(self.learning_rate)
            self.n_estimators = int(self.n_estimators)
            self.subsample = float(self.subsample)
            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
            if self.max_depth == "None":
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            num_features = X.shape[1]
            max_features = int(
                float(self.max_features) * (np.log(num_features) + 1))
            # Use at most half of the features
            max_features = max(1, min(int(X.shape[1] / 2), max_features))
            if self.max_leaf_nodes == "None":
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.verbose = int(self.verbose)

            self.estimator = sklearn.ensemble.GradientBoostingClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                n_estimators=0,
                subsample=self.subsample,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                max_features=max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                init=self.init,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=True,
            )

        tmp = self.estimator  # TODO copy ?
        tmp.n_estimators += n_iter
        tmp.fit(X, y, sample_weight=sample_weight)
        self.estimator = tmp
        # Apparently this if is necessary
        if self.estimator.n_estimators >= self.n_estimators:
            self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        #loss = cs.add_hyperparameter(Constant("loss", "deviance"))
        learning_rate = cs.add_hyperparameter(UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default=0.1, log=True))
        n_estimators = cs.add_hyperparameter(UniformIntegerHyperparameter
            ("n_estimators", 50, 500, default=100))
        max_depth = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default=3))
        min_samples_split = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="min_samples_split", lower=2, upper=20, default=2, log=False))
        min_samples_leaf = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default=1, log=False))
        #min_weight_fraction_leaf = cs.add_hyperparameter(
        #    UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.))
        subsample = cs.add_hyperparameter(UniformFloatHyperparameter(
                name="subsample", lower=0.01, upper=1.0, default=1.0, log=False))
        max_features = cs.add_hyperparameter(UniformFloatHyperparameter(
            "max_features", 0.5, 5, default=1))
        #max_leaf_nodes = cs.add_hyperparameter(UnParametrizedHyperparameter(
        #    name="max_leaf_nodes", value="None"))

        return cs

