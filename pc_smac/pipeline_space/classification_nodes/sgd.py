
# The hyperparameter spaces come from https://github.com/automl/auto-sklearn

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from pc_smac.pc_smac.pipeline_space.node import Node

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