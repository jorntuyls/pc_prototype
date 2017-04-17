import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *

class ExtraTreesNode(Node):
    def __init__(self):
        self.name = "extra_rand_trees"
        self.type = "feature_preprocessor"
        # 5 Actual hyperparameters
        self.hyperparameters = {"n_estimators": 100, "criterion": "gini", "max_features": 1.0,
                                "max_depth": "None", "min_samples_split": 2, "min_samples_leaf": 1,
                                "min_weight_fraction_leaf": 0.0, "bootstrap": "False"}
        self.algorithm = ExtraTreesPreprocessor

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        fast_ica = self.algorithm(n_estimators=hyperparameters["n_estimators"], # constant
                                  criterion=hyperparameters["criterion"],
                                  max_features=hyperparameters["max_features"],
                                  max_depth=hyperparameters["max_depth"],
                                  min_samples_split=hyperparameters["min_samples_split"],
                                  min_samples_leaf=hyperparameters["min_samples_leaf"],
                                  min_weight_fraction_leaf=hyperparameters["min_weight_fraction_leaf"], # constant
                                  bootstrap=hyperparameters["bootstrap"])

        return (self.get_full_name(), fast_ica)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class ExtraTreesPreprocessor(PreprocessingAlgorithm):

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split, max_features,
                 max_leaf_nodes_or_max_depth="max_depth",
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

        self.max_leaf_nodes_or_max_depth = max_leaf_nodes_or_max_depth
        if self.max_leaf_nodes_or_max_depth == "max_depth":
            self.max_leaf_nodes = None
            if max_depth == "None":
                self.max_depth = None
            else:
                self.max_depth = int(max_depth)
                # if use_max_depth == "True":
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
        else:
            self.bootstrap = bootstrap

        self.oob_score = oob_score
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.class_weight = class_weight

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'min_samples_leaf': self.min_samples_leaf,
            'min_samples_split': self.min_samples_split,
            'max_features': self.max_features,
            'max_leaf_nodes_or_max_depth': self.max_leaf_nodes_or_max_depth,
            'bootstrap': self.bootstrap,
            'max_leaf_nodes': self.max_leaf_nodes,
            'max_depth': str(self.max_depth),
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'class_weight': self.class_weight
        }

    def fit(self, X, Y, sample_weight=None):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        num_features = X.shape[1]
        max_features = int(
            float(self.max_features) * (np.log(num_features) + 1))
        # Use at most half of the features
        max_features = max(1, min(int(X.shape[1] / 2), max_features))
        estimator = ExtraTreesClassifier(
            n_estimators=self.n_estimators, criterion=self.criterion,
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap,
            max_features=max_features, max_leaf_nodes=self.max_leaf_nodes,
            oob_score=self.oob_score, n_jobs=self.n_jobs, verbose=self.verbose,
            random_state=self.random_state, class_weight=self.class_weight)
        estimator.fit(X, Y, sample_weight=sample_weight)
        self.preprocessor = SelectFromModel(estimator=estimator,
                                            threshold='mean',
                                            prefit=True)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ETC',
                'name': 'Extra Trees Classifier Preprocessing',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        #n_estimators = cs.add_hyperparameter(Constant("n_estimators", 100))
        criterion = cs.add_hyperparameter(CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default="gini"))
        max_features = cs.add_hyperparameter(UniformFloatHyperparameter(
            "max_features", 0.5, 5, default=1))

        max_depth = cs.add_hyperparameter(
            UnParametrizedHyperparameter(name="max_depth", value="None"))

        min_samples_split = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default=2))
        min_samples_leaf = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default=1))
        #min_weight_fraction_leaf = cs.add_hyperparameter(Constant(
        #    'min_weight_fraction_leaf', 0.))

        bootstrap = cs.add_hyperparameter(CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default="False"))

        return cs
