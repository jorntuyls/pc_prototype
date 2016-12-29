
__author__ = 'jorntuyls'

import sys

from data_loader import DataLoader
from config_space_builder import ConfigSpaceBuilder
from pipeline_space import PipelineSpace, TestPreprocessingStep, TestClassificationStep
from pipeline_runner import PipelineRunner, PipelineTester
from smbo_builder import SMBOBuilder
from metrics import BACMetric

from smac.runhistory.runhistory import RunHistory
from smac.smbo.objective import average_cost


data_path = "/Users/jorntuyls/Documents/workspaces/thesis/data/554_bac"

class Driver:

    def __init__(self, data_path):
        self.data_loader = DataLoader(data_path)
        self.data = self.data_loader.get_data()

        self.pipeline_space = self.build_pipeline_space()

        self.cs_builder = ConfigSpaceBuilder(self.pipeline_space)
        config_space = self.cs_builder.build_config_space()

        # Build runhistory
        runhistory = RunHistory(average_cost)

        self.tae_runner = PipelineRunner(self.data, self.pipeline_space, runhistory, downsampling=2000)

        # Build SMBO object
        smbo_builder = SMBOBuilder()
        self.smbo = smbo_builder.build_pc_smbo(
                                config_space=config_space,
                                tae_runner=self.tae_runner,
                                runhistory=runhistory,
                                aggregate_func=average_cost,
                                wallclock_limit=10)

    def run(self):
        incumbent = self.smbo.run()
        #tester = PipelineTester(self.data, self.pipeline_space, BACMetric())
        #score = tester.run(incumbent)
        #print("Incumbent: {}, Score: {}".format(incumbent, score))
        #return incumbent, score

    def build_pipeline_space(self):
        ps = PipelineSpace()
        tp = TestPreprocessingStep()
        tc = TestClassificationStep()
        ps.add_pipeline_steps([tp, tc])
        return ps




if __name__ == "__main__":
    sys
    d = Driver(data_path)
    d.run()
