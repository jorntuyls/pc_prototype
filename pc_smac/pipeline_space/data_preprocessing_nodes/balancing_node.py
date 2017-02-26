
import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *


class BalancingNode(Node):

    def __init__(self):
        self.name = "balancer"
        self.type = "balancing"
        self.hyperparameters = {"strategy": "none"}
        self.algorithm = Balancing

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        balancing = self.algorithm(strategy=self.hyperparameters["strategy"])
        return (self.get_full_name(), balancing)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)



class Balancing(PreprocessingAlgorithm):
    def __init__(self, strategy='none', random_state=None):
        self.strategy = strategy
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'random_state': self.random_state
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_weights(self, Y, classifier, preprocessor, init_params, fit_params):
        if init_params is None:
            init_params = {}

        if fit_params is None:
            fit_params = {}

        # Classifiers which require sample weights:
        # We can have adaboost in here, because in the fit method,
        # the sample weights are normalized:
        # https://github.com/scikit-learn/scikit-learn/blob/0.15.X/sklearn/ensemble/weight_boosting.py#L121
        # Have RF and ET in here because they emit a warning if class_weights
        #  are used together with warmstarts
        clf_ = ['adaboost', 'gradient_boosting', 'random_forest',
                'extra_trees', 'sgd']
        pre_ = []
        if classifier in clf_ or preprocessor in pre_:
            if len(Y.shape) > 1:
                offsets = [2 ** i for i in range(Y.shape[1])]
                Y_ = np.sum(Y * offsets, axis=1)
            else:
                Y_ = Y

            unique, counts = np.unique(Y_, return_counts=True)
            cw = 1. / counts
            cw = cw / np.mean(cw)

            sample_weights = np.ones(Y_.shape)

            for i, ue in enumerate(unique):
                mask = Y_ == ue
                sample_weights[mask] *= cw[i]

            if classifier in clf_:
                fit_params['classifier:sample_weight'] = sample_weights
            if preprocessor in pre_:
                fit_params['preprocessor:sample_weight'] = sample_weights

        # Classifiers which can adjust sample weights themselves via the
        # argument `class_weight`
        clf_ = ['decision_tree', 'liblinear_svc',
                'libsvm_svc']
        pre_ = ['liblinear_svc_preprocessor',
                'extra_trees_preproc_for_classification']
        if classifier in clf_:
            init_params['classifier:class_weight'] = 'auto'
        if preprocessor in pre_:
            init_params['preprocessor:class_weight'] = 'auto'

        clf_ = ['ridge']
        if classifier in clf_:
            class_weights = {}

            unique, counts = np.unique(Y, return_counts=True)
            cw = 1. / counts
            cw = cw / np.mean(cw)

            for i, ue in enumerate(unique):
                class_weights[ue] = cw[i]

            if classifier in clf_:
                init_params['classifier:class_weight'] = class_weights

        return init_params, fit_params

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter(
            "strategy", ["none", "weighting"], default="none")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Balancing',
                'name': 'Balancing Imbalanced Class Distributions',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}



