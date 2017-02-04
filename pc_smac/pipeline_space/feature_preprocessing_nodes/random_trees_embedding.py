from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, Constant

from pc_smac.pc_smac.pipeline_space.node import Node, PreprocessingAlgorithm
from pc_smac.pc_smac.utils.constants import *

class RandomTreesEmbeddingNode(Node):

    def __init__(self):
        self.name = "random_trees_embedding"
        self.type = "feature_preprocessor"
        self.hyperparameters = {"n_estimators": 10, "max_depth": 5, "min_samples_split": 2,
                                "min_samples_leaf": 1, "min_weight_fraction_leaf": 1.0, "max_leaf_nodes":"None"}
        self.algorithm = RandomTreesEmbedding

    def initialize_algorithm(self, hyperparameters):
        hyperparameters = self.initialize_hyperparameters(hyperparameters)
        print("TEST")
        random_trees_embedding = self.algorithm(n_estimators=hyperparameters["n_estimators"],
                                             max_depth=hyperparameters["max_depth"],
                                             min_samples_split=hyperparameters["min_samples_split"],
                                             min_samples_leaf=hyperparameters["min_samples_leaf"],
                                             min_weight_fraction_leaf=hyperparameters["min_weight_fraction_leaf"],
                                             max_leaf_nodes="None")

        return (self.get_full_name(), random_trees_embedding)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        return self.algorithm.get_hyperparameter_search_space(dataset_properties=dataset_properties)

    def get_properties(self, dataset_properties=None):
        return self.algorithm.get_properties(dataset_properties=dataset_properties)

class RandomTreesEmbedding(PreprocessingAlgorithm):

    def __init__(self, n_estimators, max_depth, min_samples_split,
                 min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes,
                 sparse_output=True, n_jobs=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.sparse_output = sparse_output
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, Y=None):
        import sklearn.ensemble

        if self.max_depth == "None":
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)
        if self.max_leaf_nodes == "None" or self.max_leaf_nodes == None:
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        self.preprocessor = sklearn.ensemble.RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            sparse_output=self.sparse_output,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RandomTreesEmbedding',
                'name': 'Random Trees Embedding',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (SPARSE, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_estimators = UniformIntegerHyperparameter(name="n_estimators",
                                                    lower=10, upper=100,
                                                    default=10)
        max_depth = UniformIntegerHyperparameter(name="max_depth",
                                                 lower=2, upper=10,
                                                 default=5)
        min_samples_split = UniformIntegerHyperparameter(name="min_samples_split",
                                                         lower=2, upper=20,
                                                         default=2)
        min_samples_leaf = UniformIntegerHyperparameter(name="min_samples_leaf",
                                                        lower=1, upper=20,
                                                        default=1)
        #min_weight_fraction_leaf = Constant('min_weight_fraction_leaf', 1.0)
        max_leaf_nodes = UnParametrizedHyperparameter(name="max_leaf_nodes",
                                                      value="None")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_estimators)
        cs.add_hyperparameter(max_depth)
        cs.add_hyperparameter(min_samples_split)
        cs.add_hyperparameter(min_samples_leaf)
        #cs.add_hyperparameter(min_weight_fraction_leaf)
        cs.add_hyperparameter(max_leaf_nodes)
        return cs
