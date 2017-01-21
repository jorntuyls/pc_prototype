
from pipeline_space.node import KernelPcaNode, SGDNode


class PipelineStep(object):

    def __init__(self, name, nodes):
        self.step_name = name
        self.nodes = nodes

    def get_name(self):
        return self.step_name

    def get_nodes(self):
        return self.nodes

    def get_node_names(self):
        return [node.get_name() for node in self.get_nodes()]

    def initialize_algorithm(self, node_name, hyperparameters):
        node = self._get_node(node_name)
        return node.initialize_algorithm(hyperparameters)

    #### Internal methods ####
    def _get_node(self, node_name):
        temp = [node for node in self.get_nodes() if node.get_name() == node_name]
        return temp[0]




class TestPreprocessingStep(PipelineStep):

    def __init__(self):
        name = "feature_preprocessor"
        nodes = {KernelPcaNode()}
        super(TestPreprocessingStep, self).__init__(name, nodes)

class TestClassificationStep(PipelineStep):

    def __init__(self):
        name = "classifier"
        nodes = {SGDNode()}
        super(TestClassificationStep, self).__init__(name, nodes)
