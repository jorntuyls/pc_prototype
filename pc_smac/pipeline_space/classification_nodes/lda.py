



# The algorithm and hyperparameter spaces come from https://github.com/automl/auto-sklearn


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

from pc_smac.pc_smac.pipeline_space.node import Node, ClassificationAlgorithm
from pc_smac.pc_smac.utils.constants import *
from pc_smac.pc_smac.utils.util_implementations import softmax

class LDANode(Node):

    def __init__(self):
        self.name = "lda"
        self.type = "classifier"
        self.hyperparameters = {"shrinkage": "None", "shrinkage_factor": 0.5, "n_components": 10, "tol": 1e-4}
        self.algorithm = LDA

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        lda = self.algorithm(shrinkage=hyperparameters["shrinkage"],
                             n_components=hyperparameters["n_components"],
                             tol=hyperparameters["tol"],
                             shrinkage_factor=hyperparameters["shrinkage_factor"])

        return (self.get_full_name(), lda)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class LDA(ClassificationAlgorithm):
    def __init__(self, shrinkage, n_components, tol, shrinkage_factor=0.5,
        random_state=None):
        self.shrinkage = shrinkage
        self.n_components = n_components
        self.tol = tol
        self.shrinkage_factor = shrinkage_factor
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.discriminant_analysis
        import sklearn.multiclass

        if self.shrinkage == "None":
            self.shrinkage = None
            solver = 'svd'
        elif self.shrinkage == "auto":
            solver = 'lsqr'
        elif self.shrinkage == "manual":
            self.shrinkage = float(self.shrinkage_factor)
            solver = 'lsqr'
        else:
            raise ValueError(self.shrinkage)

        self.n_components = int(self.n_components)
        self.tol = float(self.tol)

        estimator = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            n_components=self.n_components, shrinkage=self.shrinkage,
            tol=self.tol, solver=solver)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        df = self.estimator.predict_proba(X)
        return softmax(df)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LDA',
                'name': 'Linear Discriminant Analysis',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        shrinkage = cs.add_hyperparameter(CategoricalHyperparameter(
            "shrinkage", ["None", "auto", "manual"], default="None"))
        shrinkage_factor = cs.add_hyperparameter(UniformFloatHyperparameter(
            "shrinkage_factor", 0., 1., 0.5))
        n_components = cs.add_hyperparameter(UniformIntegerHyperparameter(
            'n_components', 1, 250, default=10))
        tol = cs.add_hyperparameter(UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default=1e-4, log=True))

        cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))
        return cs
