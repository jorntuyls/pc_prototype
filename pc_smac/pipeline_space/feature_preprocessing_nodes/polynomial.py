

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter


from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *

class PolynomialFeaturesNode(Node):

    def __init__(self):
        self.name = "polynomial_features"
        self.type = "feature_preprocessor"
        self.hyperparameters = {"degree": 2, "interaction_only": "False", "include_bias": "True"}
        self.algorithm = PolynomialFeatures

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        polynomial_features = self.algorithm(degree=hyperparameters["degree"],
                                             interaction_only=hyperparameters["interaction_only"],
                                             include_bias=hyperparameters["include_bias"])

        return (self.get_full_name(), polynomial_features)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class PolynomialFeatures(PreprocessingAlgorithm):
    def __init__(self, degree, interaction_only, include_bias, random_state=None):
        self.degree = int(degree)
        self.interaction_only = interaction_only.lower() == 'true'
        self.include_bias = include_bias.lower() == 'true'
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.preprocessing

        self.preprocessor = sklearn.preprocessing.PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only,
            include_bias=self.include_bias)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PolynomialFeatures',
                'name': 'PolynomialFeatures',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # More than degree 3 is too expensive!
        degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
        interaction_only = CategoricalHyperparameter("interaction_only",
                                                     ["False", "True"], "False")
        include_bias = CategoricalHyperparameter("include_bias",
                                                 ["True", "False"], "True")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(degree)
        cs.add_hyperparameter(interaction_only)
        cs.add_hyperparameter(include_bias)

        return cs

