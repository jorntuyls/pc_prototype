
from ConfigSpace.configuration_space import ConfigurationSpace
from pc_smac.pc_smac.pipeline_space.node import PreprocessingAlgorithm

class Rescaling(PreprocessingAlgorithm):

    def get_params(self, deep=True):
        return {}

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs