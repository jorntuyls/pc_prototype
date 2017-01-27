

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction

from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingNode
from pc_smac.pc_smac.utils.constants import *

class LinearSVMNode(Node):

    def __init__(self):
        self.name = "linear_svm"
        self.type = "feature_preprocessor"
        self.hyperparameters = {"penalty": "l1", "loss": "squared_hinge", "dual": "False", "tol": 1e-4,
                                "C": 1.0, "multi_class": "ovr", "fit_intercept": "True", "intercept_scaling": 1}
        self.algorithm = LibLinear_Preprocessor

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        linear_svm = self.algorithm(penalty=hyperparameters["penalty"],
                                    loss=hyperparameters["loss"], # real hp
                                    dual=hyperparameters["dual"],
                                    tol=hyperparameters["tol"], # real hp
                                    C=hyperparameters["C"], # real hp
                                    multi_class=hyperparameters["multi_class"],
                                    fit_intercept=hyperparameters["fit_intercept"],
                                    intercept_scaling=hyperparameters["intercet_scaling"])

        return (self.get_full_name(), linear_svm)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)


class LibLinear_Preprocessor(PreprocessingNode):
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, penalty, loss, dual, tol, C, multi_class,
                 fit_intercept, intercept_scaling, class_weight=None,
                 random_state=None):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.svm
        from sklearn.feature_selection import SelectFromModel

        self.C = float(self.C)
        self.tol = float(self.tol)

        self.dual = self.dual == 'True'
        self.fit_intercept = self.fit_intercept == 'True'
        self.intercept_scaling = float(self.intercept_scaling)

        if self.class_weight == "None":
            self.class_weight = None

        estimator = sklearn.svm.LinearSVC(penalty=self.penalty,
                                          loss=self.loss,
                                          dual=self.dual,
                                          tol=self.tol,
                                          C=self.C,
                                          class_weight=self.class_weight,
                                          fit_intercept=self.fit_intercept,
                                          intercept_scaling=self.intercept_scaling,
                                          multi_class=self.multi_class,
                                          random_state=self.random_state)

        estimator.fit(X, Y)
        self.preprocessor = SelectFromModel(estimator=estimator,
                                            threshold='mean',
                                            prefit=True)

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LinearSVC Preprocessor',
                'name': 'Liblinear Support Vector Classification Preprocessing',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        penalty = cs.add_hyperparameter(Constant("penalty", "l1"))
        loss = cs.add_hyperparameter(CategoricalHyperparameter(
            "loss", ["hinge", "squared_hinge"], default="squared_hinge"))
        dual = cs.add_hyperparameter(Constant("dual", "False"))
        # This is set ad-hoc
        tol = cs.add_hyperparameter(UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default=1e-4, log=True))
        C = cs.add_hyperparameter(UniformFloatHyperparameter(
            "C", 0.03125, 32768, log=True, default=1.0))
        multi_class = cs.add_hyperparameter(Constant("multi_class", "ovr"))
        # These are set ad-hoc
        fit_intercept = cs.add_hyperparameter(Constant("fit_intercept", "True"))
        intercept_scaling = cs.add_hyperparameter(Constant(
            "intercept_scaling", 1))

        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"),
            ForbiddenEqualsClause(loss, "hinge")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        return cs
