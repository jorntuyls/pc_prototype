
import abc

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

class Node(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def get_name(self):
        return self.name

    def get_full_name(self):
        return self.type + ":" + self.name

    def get_hyperparameters(self):
        return self.hyperparameters.keys()

    def initialize_hyperparameters(self, hyperparameters):
        for hp in self.get_hyperparameters():
            if hp not in hyperparameters or hyperparameters[hp] == None:
                hyperparameters[hp] = self.hyperparameters[hp]
        return hyperparameters

    @abc.abstractmethod
    def initialize_algorithm(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_hyperparameter_space(self):
        raise NotImplementedError




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

class SGDNode(Node):

    def __init__(self):
        self.name = "sgd"
        self.type = "classifier"
        self.hyperparameters = {"loss": "log", "penalty": "l2", "alpha": 1e-4, "l1_ratio": 0.15,
                "fit_intercept": True, "n_iter": 20, "epsilon": 1e-4, "learning_rate": "optimal",
                "eta0": 0.01, "power_t": 0.25, "average": "False"}

    def initialize_algorithm(self, hyperparameters):
        from sklearn.linear_model.stochastic_gradient import SGDClassifier
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        sgd = SGDClassifier(
                            loss=hyperparameters["loss"],
                            penalty=hyperparameters["penalty"],
                            alpha=float(hyperparameters["alpha"]),
                            fit_intercept=bool(hyperparameters["fit_intercept"]),
                            n_iter=int(hyperparameters["n_iter"]),
                            learning_rate=hyperparameters["learning_rate"],
                            l1_ratio=float(hyperparameters["l1_ratio"]),
                            epsilon=float(hyperparameters["epsilon"]),
                            eta0=float(hyperparameters["eta0"]),
                            power_t=float(hyperparameters["power_t"]),
                            shuffle=True,
                            average=bool(hyperparameters["average"]),
                            random_state=None)

        return (self.get_full_name(), sgd)

    def get_hyperparameter_space(self):
        cs = ConfigurationSpace()
        loss = cs.add_hyperparameter(CategoricalHyperparameter("loss",
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            default="log"))
        penalty = cs.add_hyperparameter(CategoricalHyperparameter(
            "penalty", ["l1", "l2", "elasticnet"], default="l2"))
        alpha = cs.add_hyperparameter(UniformFloatHyperparameter(
            "alpha", 10e-7, 1e-1, log=True, default=0.0001))
        l1_ratio = cs.add_hyperparameter(UniformFloatHyperparameter(
            "l1_ratio", 1e-9, 1,  log=True, default=0.15))
        fit_intercept = cs.add_hyperparameter(UnParametrizedHyperparameter(
            "fit_intercept", "True"))
        n_iter = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "n_iter", 5, 100, log=True, default=20))
        epsilon = cs.add_hyperparameter(UniformFloatHyperparameter(
            "epsilon", 1e-5, 1e-1, default=1e-4, log=True))
        learning_rate = cs.add_hyperparameter(CategoricalHyperparameter(
            "learning_rate", ["optimal", "invscaling", "constant"],
            default="optimal"))
        eta0 = cs.add_hyperparameter(UniformFloatHyperparameter(
            "eta0", 10**-7, 0.1, default=0.01))
        power_t = cs.add_hyperparameter(UniformFloatHyperparameter(
            "power_t", 1e-5, 1, default=0.25))
        average = cs.add_hyperparameter(CategoricalHyperparameter(
            "average", ["False", "True"], default="False"))

        # TODO add passive/aggressive here, although not properly documented?
        elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
        epsilon_condition = EqualsCondition(epsilon, loss, "modified_huber")
        # eta0 seems to be always active according to the source code; when
        # learning_rate is set to optimial, eta0 is the starting value:
        # https://github.com/scikit-learn/scikit-learn/blob/0.15.X/sklearn/linear_model/sgd_fast.pyx
        #eta0_and_inv = EqualsCondition(eta0, learning_rate, "invscaling")
        #eta0_and_constant = EqualsCondition(eta0, learning_rate, "constant")
        #eta0_condition = OrConjunction(eta0_and_inv, eta0_and_constant)
        power_t_condition = EqualsCondition(power_t, learning_rate, "invscaling")

        cs.add_condition(elasticnet)
        cs.add_condition(epsilon_condition)
        cs.add_condition(power_t_condition)

        return cs
