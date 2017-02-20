
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from pc_smac.pc_smac.whitebox.whitebox import WhiteBoxFunction, CachedWhiteboxFunction

class Beale(WhiteBoxFunction):

    def __init__(self, runhistory, statistics, min_x=[0.75, 0.25], min_y=[0.5, 0.5]):
        self.min_x1 = min_x[0]
        self.min_x2 = min_x[1]
        self.min_y1 = min_y[0]
        self.min_y2 = min_y[1]
        super(Beale, self).__init__(runhistory, statistics)

    def z_func(self, x,y):
        return (1.5 - (x - 4.5) + (x - 4.5)*(y-4.5))**2 + (2.25 - (x - 4.5) + (x - 4.5)*(y - 4.5)**2)**2 + (2.625 - (x-4.5) + (x-4.5)*(y-4.5)**3)**2

    def get_config_space(self, seed=None):
        cs = ConfigurationSpace() if seed == None else ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "x", 0, 9, default=0.))
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "y", 0, 9, default=0.))
        return cs

class CachedBeale(CachedWhiteboxFunction):

    def __init__(self, runhistory, statistics, min_x=[0.75, 0.25], min_y=[0.5, 0.5]):
        self.min_x1 = min_x[0]
        self.min_x2 = min_x[1]
        self.min_y1 = min_y[0]
        self.min_y2 = min_y[1]
        super(CachedBeale, self).__init__(runhistory, statistics)

    def z_func(self, x, y):
        return (1.5 - (x - 4.5) + (x - 4.5)*(y-4.5))**2 + (2.25 - (x - 4.5) + (x - 4.5)*(y - 4.5)**2)**2 + (2.625 - (x-4.5) + (x-4.5)*(y-4.5)**3)**2

    def get_config_space(self, seed=None):
        cs = ConfigurationSpace() if seed == None else ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "x", 0, 9, default=4.5))
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "y", 0, 9, default=4.5))
        return cs
