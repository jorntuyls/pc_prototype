
#from sklearn.pipeline import Pipeline
import tempfile
import shutil

from sklearn.externals.joblib import Memory
from cached_pipeline import CachedPipeline

class PipelineBuilder:

    def __init__(self, pipeline_space):
        dr = "/Users/jorntuyls/Documents/workspaces/thesis/data/"
        self.pipeline_space = pipeline_space
        self.cachedir = tempfile.mkdtemp(dir=dr,prefix="cache_")
        print(self.cachedir)

    def build_pipeline(self, config):
        pipeline_steps = self.pipeline_space.get_pipeline_step_names()
        concrete_steps = []
        for ps in pipeline_steps:
            algo_name = config[ps]
            hyperparameters = {}
            for hp_name in config.keys():
                splt_hp_name = hp_name.split(":")
                # if parameter belongs to parameter space of algo
                if splt_hp_name[0] == algo_name and len(splt_hp_name) == 2:
                    hyperparameters[splt_hp_name[1]] = config[hp_name]
            step = self.pipeline_space.initialize_algorithm(ps, algo_name, hyperparameters)
            concrete_steps.append(step)

        return CachedPipeline(concrete_steps, memory=Memory(cachedir=self.cachedir, verbose=0))

    def clean_cache(self):
        shutil.rmtree(self.cachedir)
