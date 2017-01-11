
__author__ = 'jorntuyls'

import sys

from data_loader import DataLoader
from config_space_builder import ConfigSpaceBuilder
from pipeline_space import PipelineSpace, TestPreprocessingStep, TestClassificationStep
from pipeline_runner import PipelineRunner, CachedPipelineRunner, PipelineTester
from smbo_builder import SMBOBuilder
from pc_runhistory import PCRunHistory

from smac.runhistory.runhistory import RunHistory
from smac.smbo.objective import average_cost

from data_paths import data_path, cache_directory

class Driver:

    def __init__(self, data_path, caching=True):
        self.data_loader = DataLoader(data_path)
        self.data = self.data_loader.get_data()

        self.pipeline_space = self._build_pipeline_space()

        self.cs_builder = ConfigSpaceBuilder(self.pipeline_space)
        config_space = self.cs_builder.build_config_space()

        # Build runhistory
        runhistory = PCRunHistory(average_cost)

        # Set cache directory
        if caching:
            cache_dir = cache_directory
            self.tae_runner = CachedPipelineRunner(self.data, self.pipeline_space, runhistory,
                                            cache_directory=cache_dir,
                                            downsampling=2000)
        else:
            self.tae_runner = PipelineRunner(self.data, self.pipeline_space, runhistory,
                                             downsampling=2000)



        # Choose acquisition function
        if caching:
            acq_func_name = "pceips"
            model_target_names = ['cost','time']
        else:
            acq_func_name = "ei"
            model_target_names = ['cost']

        # Build SMBO object
        smbo_builder = SMBOBuilder()
        self.smbo = smbo_builder.build_pc_smbo(
                                config_space=config_space,
                                tae_runner=self.tae_runner,
                                runhistory=runhistory,
                                aggregate_func=average_cost,
                                acq_func_name=acq_func_name,
                                model_target_names=model_target_names,
                                wallclock_limit=720)

    def run(self):
        incumbent = self.smbo.run()
        #tester = PipelineTester(self.data, self.pipeline_space)
        #score = tester.run(incumbent)
        #print("Incumbent: {}, Score: {}".format(incumbent, score))
        #return incumbent, score

    def _build_pipeline_space(self):
        ps = PipelineSpace()
        tp = TestPreprocessingStep()
        tc = TestClassificationStep()
        ps.add_pipeline_steps([tp, tc])
        return ps




if __name__ == "__main__":
    sys
    d = Driver(data_path, caching=True)
    d.run()
