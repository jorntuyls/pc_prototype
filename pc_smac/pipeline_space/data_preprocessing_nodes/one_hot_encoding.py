import numpy as np
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from pc_smac.pc_smac.pipeline_space.algorithms.one_hot_encoding import OneHotEncoderImplementation
from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *

class OneHotEncodeingNode(Node):

    def __init__(self):
        self.name = "one_hot_encoding"
        self.type = "one_hot_encoder"
        self.hyperparameters = {"use_minimum_fraction": "True", "minimum_fraction": 0.01}
        self.algorithm = OneHotEncoder

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        one_hot_encoder = self.algorithm(use_minimum_fraction=hyperparameters["use_minimum_fraction"],
                                        minimum_fraction=hyperparameters["minimum_fraction"])
        return (self.get_full_name(), one_hot_encoder)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class OneHotEncoder(PreprocessingAlgorithm):
    def __init__(self, use_minimum_fraction=True, minimum_fraction=0.01,
                 categorical_features=None, random_state=None):
        # TODO pay attention to the cases when a copy is made (CSR matrices)
        self.use_minimum_fraction = use_minimum_fraction
        self.minimum_fraction = minimum_fraction
        self.categorical_features = categorical_features

    def get_params(self, deep=True):
        return {
            'use_minimum_fraction': self.use_minimum_fraction,
            'minimum_fraction': self.minimum_fraction,
            'categorical_features': self.categorical_features
        }

    def fit(self, X, y=None):
        if self.use_minimum_fraction is None or \
                self.use_minimum_fraction is False or \
                (isinstance(self.use_minimum_fraction, str) and
                 self.use_minimum_fraction.lower() == 'false'):
            self.minimum_fraction = None
        else:
            self.minimum_fraction = float(self.minimum_fraction)

        if self.categorical_features is None:
            categorical_features = []
        else:
            categorical_features = self.categorical_features

        self.preprocessor = OneHotEncoderImplementation(minimum_fraction=self.minimum_fraction,
                           categorical_features=categorical_features)

        self.preprocessor = self.preprocessor.fit(X)
        return self

    def transform(self, X):
        import scipy.sparse

        is_sparse = scipy.sparse.issparse(X)
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X)
        if is_sparse:
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            return X.toarray()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': '1Hot',
                'name': 'One Hot Encoder',
                'handles_missing_values': True,
                'handles_nominal_values': True,
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
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        use_minimum_fraction = cs.add_hyperparameter(CategoricalHyperparameter(
            "use_minimum_fraction", ["True", "False"], default="True"))
        minimum_fraction = cs.add_hyperparameter(UniformFloatHyperparameter(
            "minimum_fraction", lower=.0001, upper=0.5, default=0.01, log=True))
        cs.add_condition(EqualsCondition(minimum_fraction,
                                         use_minimum_fraction, 'True'))
        return cs


