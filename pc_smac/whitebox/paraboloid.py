

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from pc_smac.pc_smac.whitebox.whitebox import WhiteBoxFunction, CachedWhiteboxFunction

class Paraboloid(WhiteBoxFunction):

    def __init__(self, runhistory, statistics, min_x=0.75, min_y=0.5):
        self.min_x = min_x
        self.min_y = min_y
        super(Paraboloid, self).__init__(runhistory, statistics)

    def z_func(self, x,y):
        return (x - self.min_x)**2/0.1**2 + (y - self.min_y)**2/0.1**2

    def get_config_space(self, seed=None):
        cs = ConfigurationSpace() if seed == None else ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "preprocessor:x", 0.0, 1.0, default=0.5))
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "classifier:y", 0.0, 1.0, default=0.5))
        return cs

class CachedParaboloid(CachedWhiteboxFunction):

    def __init__(self, runhistory, statistics, min_x=0.75, min_y=0.5):
        self.min_x = min_x
        self.min_y = min_y
        super(CachedParaboloid, self).__init__(runhistory, statistics)

    def z_func(self, x, y):
        return (x - self.min_x) ** 2 / 0.1 ** 2 + (y - self.min_y) ** 2 / 0.1 ** 2

    def get_config_space(self, seed=None):
        cs = ConfigurationSpace() if seed == None else ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "preprocessor:x", 0.0, 1.0, default=0.5))
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "classifier:y", 0.0, 1.0, default=0.5))
        return cs


class Paraboloid2Minima(WhiteBoxFunction):

    def __init__(self, runhistory, statistics, min_x=[0.75, 0.25], min_y=[0.5, 0.5]):
        self.min_x1 = min_x[0]
        self.min_x2 = min_x[1]
        self.min_y1 = min_y[0]
        self.min_y2 = min_y[1]
        super(Paraboloid2Minima, self).__init__(runhistory, statistics)

    def z_func(self, x,y):
        return (x - self.min_x1) ** 2 * (x - self.min_x2) ** 2 / 0.1 ** 2 + (y - self.min_y1) ** 2 * (y - self.min_y2) ** 2/ 0.1 ** 2

    def get_config_space(self, seed=None):
        cs = ConfigurationSpace() if seed == None else ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "preprocessor:x", 0.0, 1.0, default=0.5))
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "classifier:y", 0.0, 1.0, default=0.5))
        return cs

class CachedParaboloid2Minima(CachedWhiteboxFunction):

    def __init__(self, runhistory, statistics, min_x=[0.75, 0.25], min_y=[0.5, 0.5]):
        self.min_x1 = min_x[0]
        self.min_x2 = min_x[1]
        self.min_y1 = min_y[0]
        self.min_y2 = min_y[1]
        super(CachedParaboloid2Minima, self).__init__(runhistory, statistics)

    def z_func(self, x, y):
        return (x - self.min_x1) ** 2 * (x - self.min_x2) ** 2 / 0.1 ** 2 + (y - self.min_y1) ** 2 * (y - self.min_y2) ** 2/ 0.1 ** 2

    def get_config_space(self, seed=None):
        cs = ConfigurationSpace() if seed == None else ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "preprocessor:x", 0.0, 1.0, default=0.5))
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "classifier:y", 0.0, 1.0, default=0.5))
        return cs




