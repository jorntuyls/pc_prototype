

class PipelineSpace(object):

    def __init__(self):
        self.pipeline_steps = []

    def add_pipeline_step(self, pipeline_step):
        self.pipeline_steps.append(pipeline_step)

    def add_pipeline_steps(self, pipeline_steps):
        self.pipeline_steps = self.pipeline_steps + pipeline_steps

    def get_pipeline_steps(self):
        return self.pipeline_steps

    def get_pipeline_steps_names_and_objects(self):
        """

        Returns
        -------
            name for each pipeline step in this pipeline space (e.g. 'feature_preprocessor', 'classifier')
        """
        return [(ps.get_name(), ps) for ps in self.get_pipeline_steps()]

    def get_pipeline_step_names(self):
        return [ps.get_name() for ps in self.get_pipeline_steps()]

    def initialize_algorithm(self, pipeline_step_name, node_name, hyperparameters):
        """

        Parameters
        ----------
        pipeline_step_name:     the name of the pipeline step in which we want to initialize the algorithm
        node_name:              the name of the node where we want to initialize the algorithm
        hyperparameters:        the hyperparameters for the node's algorithm

        Returns
        -------
        a tuple (full algorithm name, initialized algorithm)

        """
        ps = self.get_pipeline_step(pipeline_step_name)
        return ps.initialize_algorithm(node_name, hyperparameters)

    def get_pipeline_step(self, name):
        temp = [ps for ps in self.get_pipeline_steps() if ps.get_name() == name]
        return temp[0]
