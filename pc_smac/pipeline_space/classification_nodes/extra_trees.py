


# The algorithm and hyperparameter spaces come from https://github.com/automl/auto-sklearn


import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from pc_smac.pc_smac.pipeline_space.node import Node, ClassificationAlgorithm
from pc_smac.pc_smac.utils.constants import *
from pc_smac.pc_smac.utils.util_implementations import convert_multioutput_multiclass_to_multilabel


class ExtraTreesClassifierNode(Node):

    def __init__(self):
        self.name = "extra_trees"
        self.type = "classifier"
        self.hyperparameters = {"n_estimators": 100, "criterion": "gini", "max_features": 1, "max_depth": "None",
                                "min_samples_split": 2, "min_samples_leaf": 1, "min_weight_fraction_leaf": 0.,
                                "bootstrap": "False"}
        self.algorithm = ExtraTreesClassifier

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        extra_trees = self.algorithm(n_estimators=hyperparameters["n_estimators"],
                                     criterion=hyperparameters["criterion"], # actual hp
                                     min_samples_leaf=hyperparameters["min_samples_leaf"], # actual hp
                                     min_samples_split=hyperparameters["min_samples_split"], # actual hp
                                     max_features=hyperparameters["max_features"], # actual hp
                                     max_depth=hyperparameters["max_depth"],
                                     min_weight_fraction_leaf=hyperparameters["min_weight_fraction_leaf"],
                                     bootstrap=hyperparameters["bootstrap"] # actual hp
                                     )

        return (self.get_full_name(), extra_trees)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)





class ExtraTreesClassifier(ClassificationAlgorithm):

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split,  max_features, max_leaf_nodes_or_max_depth="max_depth",
                 bootstrap=False, max_leaf_nodes=None, max_depth="None",
                 min_weight_fraction_leaf=0.0,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 class_weight=None):

        self.n_estimators = int(n_estimators)
        self.estimator_increment = 10
        if criterion not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): "
                             "%s" % criterion)
        self.criterion = criterion

        if max_leaf_nodes_or_max_depth == "max_depth":
            self.max_leaf_nodes = None
            if max_depth == "None":
                self.max_depth = None
            else:
                self.max_depth = int(max_depth)
            #if use_max_depth == "True":
            #    self.max_depth = int(max_depth)
            #elif use_max_depth == "False":
            #    self.max_depth = None
        else:
            if max_leaf_nodes == "None":
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(max_leaf_nodes)
            self.max_depth = None

        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)

        self.max_features = float(max_features)

        if bootstrap == "True":
            self.bootstrap = True
        elif bootstrap == "False":
            self.bootstrap = False

        self.oob_score = oob_score
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.class_weight = class_weight
        self.estimator = None

    def fit(self, X, y, sample_weight=None, refit=False):
        if self.estimator is None or refit:
            self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight,
                               refit=refit)

        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight)
        return self

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        from sklearn.ensemble import ExtraTreesClassifier as ETC

        if refit:
            self.estimator = None

        if self.estimator is None:
            num_features = X.shape[1]
            max_features = int(
                float(self.max_features) * (np.log(num_features) + 1))
            # Use at most half of the features
            max_features = max(1, min(int(X.shape[1] / 2), max_features))
            self.estimator = ETC(
                n_estimators=0, criterion=self.criterion,
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap,
                max_features=max_features, max_leaf_nodes=self.max_leaf_nodes,
                oob_score=self.oob_score, n_jobs=self.n_jobs, verbose=self.verbose,
                random_state=self.random_state,
                class_weight=self.class_weight,
                warm_start=True
            )

        tmp = self.estimator  # TODO copy ?
        tmp.n_estimators += n_iter
        tmp.fit(X, y, sample_weight=sample_weight)
        self.estimator = tmp
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ET',
                'name': 'Extra Trees Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # n_estimators = cs.add_hyperparameter(Constant("n_estimators", 100))
        criterion = cs.add_hyperparameter(CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default="gini"))
        max_features = cs.add_hyperparameter(UniformFloatHyperparameter(
            "max_features", 0.5, 5, default=1))

        # max_depth = cs.add_hyperparameter(
        #    UnParametrizedHyperparameter(name="max_depth", value="None"))

        min_samples_split = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default=2))
        min_samples_leaf = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default=1))
        # min_weight_fraction_leaf = cs.add_hyperparameter(Constant(
        #    'min_weight_fraction_leaf', 0.))

        bootstrap = cs.add_hyperparameter(CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default="False"))

        return cs
