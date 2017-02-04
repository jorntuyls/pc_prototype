from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *

class RandomKitchenSinksNode(Node):

    def __init__(self):
        self.name = "kitchen_sinks"
        self.type = "feature_preprocessor"
        self.hyperparameters = {"gamma": 1.0, "n_components": 100}
        self.algorithm = RandomKitchenSinks

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        rand_kitchen_sinks = self.algorithm(gamma=hyperparameters["gamma"],
                                       n_components=hyperparameters["n_components"],
                                       random_state=None)

        return (self.get_full_name(), rand_kitchen_sinks)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class RandomKitchenSinks(PreprocessingAlgorithm):

    def __init__(self, gamma, n_components, random_state=None):
        """ Parameters:
        gamma: float
               Parameter of the rbf kernel to be approximated exp(-gamma * x^2)

        n_components: int
               Number of components (output dimensionality) used to approximate the kernel
        """
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, Y=None):
        import sklearn.kernel_approximation

        self.preprocessor = sklearn.kernel_approximation.RBFSampler(
            self.gamma, self.n_components, self.random_state)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KitchenSink',
                'name': 'Random Kitchen Sinks',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT, UNSIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        gamma = UniformFloatHyperparameter(
            "gamma", 0.3, 2., default=1.0)
        n_components = UniformIntegerHyperparameter(
            "n_components", 50, 10000, default=100, log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(n_components)
        return cs



