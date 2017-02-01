

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
        return [(ps.get_name(), ps) for ps in self.get_pipeline_steps()]

    def get_pipeline_step_names(self):
        return [ps.get_name() for ps in self.get_pipeline_steps()]

    def initialize_algorithm(self, pipeline_step_name, node_name, hyperparameters):
        ps = self.get_pipeline_step(pipeline_step_name)
        return ps.initialize_algorithm(node_name, hyperparameters)

    def get_pipeline_step(self, name):
        temp = [ps for ps in self.get_pipeline_steps() if ps.get_name() == name]
        return temp[0]
