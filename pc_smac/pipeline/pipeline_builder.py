
import os
import tempfile
import shutil

from sklearn.externals.joblib import Memory
from pc_smac.pc_smac.pipeline.cached_pipeline import CachedPipeline
from pc_smac.pc_smac.pipeline.pipeline import OwnPipeline

class PipelineBuilder:

    def __init__(self, pipeline_space, caching, cache_directory=None):
        if (caching == False) and (cache_directory != None):
            raise ValueError("Caching is disabled but a cache directory is given!")

        self.caching = caching
        self.pipeline_space = pipeline_space
        if self.caching and cache_directory:
            self.cachedir = tempfile.mkdtemp(dir=cache_directory, prefix="cache_")
            print(self.cachedir)
        elif self.caching:
            self.cachedir = tempfile.mkdtemp(prefix="cache_")
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

        if self.caching:
            return CachedPipeline(concrete_steps, memory=Memory(cachedir=self.cachedir, verbose=0))
        return OwnPipeline(concrete_steps)

    def clean_cache(self):
        if self.caching == True and os.path.exists(self.cachedir):
            shutil.rmtree(self.cachedir)
        else:
            raise ValueError("There is no cache")
