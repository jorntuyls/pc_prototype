
# Parts are copied from Auto-sklearn: https://github.com/automl/auto-sklearn

import abc

from constants import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION

#### CLASSIFICATION METRICS ####

class Metric(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def calculate_score(self, solution, prediction, task):
        pass

class BACMetric(Metric):

    def __init__(self):
        pass

    # !! for classification task
    def calculate_score(self, solution, prediction):
        a = float(len([i for i in range(0,len(solution)) if (solution[i] == 0 and prediction[i] == 0)]))
        b = float(len([i for i in range(0,len(solution)) if (solution[i] == 0 and prediction[i] == 1)]))
        c = float(len([i for i in range(0,len(solution)) if (solution[i] == 1 and prediction[i] == 0)]))
        d = float(len([i for i in range(0,len(solution)) if (solution[i] == 1 and prediction[i] == 1)]))
        print(a,b,c,d)
        return 0.5*(b/(a+b) + c/(c+d))
