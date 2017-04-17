
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *

class FastICANode(Node):
    def __init__(self):
        self.name = "fast_ica"
        self.type = "feature_preprocessor"
        self.hyperparameters = {"n_components": 100, "algorithm": "parallel", "whiten": "False",
                                "fun": "logcosh"}
        self.algorithm = FastICA

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        fast_ica = self.algorithm(n_components=hyperparameters["n_components"],
                                           algorithm=hyperparameters["algorithm"],
                                           fun=hyperparameters["fun"],
                                           whiten=hyperparameters["whiten"],
                                           random_state=None)

        return (self.get_full_name(), fast_ica)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class FastICA(PreprocessingAlgorithm):
    def __init__(self, algorithm, whiten, fun, n_components=None,
                 random_state=None):
        self.n_components = None if n_components is None else int(n_components)
        self.algorithm = algorithm
        self.whiten_original = whiten
        self.whiten = whiten == 'True'
        self.fun = fun
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            'n_components': self.n_components,
            'algorithm': self.algorithm,
            'whiten': self.whiten_original,
            'fun': self.fun,
            'random_state': self.random_state
        }

    def fit(self, X, Y=None):
        import sklearn.decomposition

        self.preprocessor = sklearn.decomposition.FastICA(
            n_components=self.n_components, algorithm=self.algorithm,
            fun=self.fun, whiten=self.whiten, random_state=self.random_state
        )
        # Make the RuntimeWarning an Exception!
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message='array must not contain infs or NaNs')
            try:
                self.preprocessor.fit(X)
            except ValueError as e:
                if 'array must not contain infs or NaNs' in e.args[0]:
                    raise ValueError("Bug in scikit-learn: https://github.com/scikit-learn/scikit-learn/pull/2738")

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FastICA',
                'name': 'Fast Independent Component Analysis',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT, UNSIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_components = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "n_components", 10, 2000, default=100))
        algorithm = cs.add_hyperparameter(CategoricalHyperparameter('algorithm',
            ['parallel', 'deflation'], 'parallel'))
        whiten = cs.add_hyperparameter(CategoricalHyperparameter('whiten',
            ['False', 'True'], 'False'))
        fun = cs.add_hyperparameter(CategoricalHyperparameter(
            'fun', ['logcosh', 'exp', 'cube'], 'logcosh'))

        cs.add_condition(EqualsCondition(n_components, whiten, "True"))

        return cs