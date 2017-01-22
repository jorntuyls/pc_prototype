
# The hyperparameter spaces come from https://github.com/automl/auto-sklearn

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from pc_prototype.pc_smac.pipeline_space.node import Node

class KernelPcaNode(Node):

    def __init__(self):
        self.name = "kernel_pca"
        self.type = "feature_preprocessor"
        self.hyperparameters = {"n_components": 100, "kernel": "rbf", "degree": 3,
                            "gamma": 1.0, "coef0": 0.0}

    def initialize_algorithm(self, hyperparameters):
        from sklearn.decomposition import KernelPCA
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        kernel_pca = KernelPCA(
                        n_components=hyperparameters["n_components"],
                        kernel=hyperparameters["kernel"],
                        gamma=float(hyperparameters["gamma"]),
                        degree=int(hyperparameters["degree"]),
                        coef0=float(hyperparameters["coef0"]),
                        remove_zero_eig=True)

        return (self.get_full_name(), kernel_pca)

    def get_hyperparameter_space(self):
        cs = ConfigurationSpace()
        n_components = UniformIntegerHyperparameter(
            "n_components", 10, 2000, default=100)
        kernel = CategoricalHyperparameter('kernel',
                                           ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf')
        degree = UniformIntegerHyperparameter('degree', 2, 5, 3)
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                           log=True, default=1.0)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default=0)

        cs.add_hyperparameter(n_components)
        cs.add_hyperparameter(kernel)
        cs.add_hyperparameter(degree)
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(coef0)

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["poly", "rbf"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)
        cs.add_condition(gamma_condition)

        return cs
