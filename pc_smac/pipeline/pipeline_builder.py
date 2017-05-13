
import os
import tempfile
import shutil

from sklearn.externals.joblib import Memory
from pc_smac.pc_smac.pipeline.cached_pipeline import CachedPipeline
from pc_smac.pc_smac.pipeline.pipeline import OwnPipeline

class PipelineBuilder:

    def __init__(self, pipeline_space, caching, cache_directory=None, min_runtime_for_caching=1):
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

        self.min_runtime_for_caching = min_runtime_for_caching

    def build_pipeline(self, config, run_instance=None):
        # pipeline_steps is a list of pipeline step names (e.g. feature_preprocessor, classifier)
        pipeline_steps = self.pipeline_space.get_pipeline_step_names()
        concrete_steps = []
        for ps in pipeline_steps:
            # TODO Remove this hardcoded ':__choice__'
            algo_name = config[ps + ':__choice__']
            hyperparameters = {}
            for hp_name in config.keys():
                splt_hp_name = hp_name.split(":")
                # the hyperparameters we are interested in are of the format:
                #   'pipelines_step_name:algorithm_name:hyperparameter'
                if splt_hp_name[0] == ps and splt_hp_name[1] == algo_name:
                    hyperparameters[splt_hp_name[2]] = config[hp_name]
            step = self.pipeline_space.initialize_algorithm(ps, algo_name, hyperparameters)
            concrete_steps.append(step)

        if self.caching:
            # TODO: Make this less hardcoded
            cached_step_names = []
            for name, step_algorithm in concrete_steps:
                if name.split(":")[0] in self.pipeline_space.get_cached_pipeline_step_names():
                    cached_step_names.append(name)
            return CachedPipeline(concrete_steps,
                                  cached_step_names=cached_step_names,
                                  memory=Memory(cachedir=self.cachedir, verbose=0),
                                  min_runtime_for_caching=self.min_runtime_for_caching,
                                  run_instance=run_instance)
        return OwnPipeline(concrete_steps)

    def clean_cache(self):
        if self.caching == True and os.path.exists(self.cachedir):
            shutil.rmtree(self.cachedir)
        else:
            raise ValueError("There is no cache")
