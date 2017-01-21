
__author__ = 'jorntuyls'

import os
import time

from utils.data_loader import DataLoader
from config_space.config_space_builder import ConfigSpaceBuilder
from pipeline_space.pipeline_space import PipelineSpace
from pipeline_space.pipeline_step import TestPreprocessingStep, TestClassificationStep
from pipeline.pipeline_runner import PipelineRunner, CachedPipelineRunner, PipelineTester
from pc_smbo.smbo_builder import SMBOBuilder
from pc_runhistory.pc_runhistory import PCRunHistory
from utils.io_utils import save_trajectory_for_plotting
from utils.statistics import Statistics

from smac.smbo.objective import average_cost
from smac.utils.io.traj_logging import TrajLogger

from data_paths import data_path, cache_directory

class Driver:

    def __init__(self, data_path):
        self.data_loader = DataLoader(data_path)

        self.pipeline_space = self._build_pipeline_space()

        self.cs_builder = ConfigSpaceBuilder(self.pipeline_space)
        self.config_space = self.cs_builder.build_config_space()

        self.output_dir = os.path.dirname(os.path.abspath(__file__)) + "/logging/"
        # TODO make names not hardcoded
        self.trajectory_path_json = os.path.dirname(os.path.abspath(__file__)) + "/logging/traj_aclib2.json"
        self.trajectory_path_csv = os.path.dirname(os.path.abspath(__file__)) + "/logging/traj_old.csv"

    def initialize(self, stamp, caching, cache_directory, wallclock_limit, downsampling, time_precision):
        # Load data
        self.data = self.data_loader.get_data()

        # Build runhistory
        # TODO Does this work correctly for non-caching?
        runhistory = PCRunHistory(average_cost)

        # Setup statistics
        info = {
            'stamp': stamp,
            'caching': caching,
            'cache_directory': cache_directory,
            'wallclock_limit': wallclock_limit,
            'downsampling': downsampling
        }
        self.statistics = Statistics(stamp, self.output_dir,
                                information=info,
                                total_runtime=wallclock_limit,
                                time_precision=time_precision)

        # Set cache directory
        if caching:
            cache_dir = cache_directory
            self.tae_runner = CachedPipelineRunner(self.data, self.pipeline_space, runhistory,
                                                   self.statistics,
                                                   cache_directory=cache_dir,
                                                   downsampling=downsampling)
        else:
            self.tae_runner = PipelineRunner(self.data, self.pipeline_space, runhistory, self.statistics,
                                             downsampling=downsampling)

        # Choose acquisition function
        if caching:
            acq_func_name = "pceips"
            model_target_names = ['cost', 'time']
        else:
            acq_func_name = "eips"
            model_target_names = ['cost', 'time']

        # Build SMBO object
        smbo_builder = SMBOBuilder()
        self.smbo = smbo_builder.build_pc_smbo(
            config_space=self.config_space,
            tae_runner=self.tae_runner,
            runhistory=runhistory,
            aggregate_func=average_cost,
            acq_func_name=acq_func_name,
            model_target_names=model_target_names,
            wallclock_limit=wallclock_limit)


    def run(self,
            stamp=time.time(),
            caching=True,
            cache_directory=None,
            wallclock_limit=3600,
            downsampling=None,
            run_counter=1,
            time_precision=10):
        # clean trajectory files
        self._clean_trajectory_files()

        # Initialize SMBO
        self.initialize(stamp=stamp,
                        caching=caching,
                        cache_directory=cache_directory,
                        wallclock_limit=wallclock_limit,
                        downsampling=downsampling,
                        time_precision=time_precision)
        # Start timer
        self.statistics.start_timer()

        # Run SMBO
        incumbent = self.smbo.run()

        # Save statistics
        self.statistics.save()

        # Read trajectory files with incumbents and retrieve test performances
        #self.trajectory = TrajLogger.read_traj_aclib_format(self.trajectory_path_json, self.config_space)
        #self.run_tests(self.trajectory, downsampling=downsampling)
        #print(self.trajectory)
        # Save new trajectory for plotting
        #save_trajectory_for_plotting(self.trajectory,
        #                                   wallclock_limit=wallclock_limit,
        #                                   plot_time=time_precision,
        #                                   caching=caching,
        #                                   run_counter=run_counter)

        return incumbent


    def run_tests(self, trajectory, downsampling=None):
        pt = PipelineTester(self.data, self.pipeline_space, downsampling=downsampling)

        for traj in trajectory:
            traj['test_performance'] = pt.get_performance(traj['incumbent'])

        return trajectory

    #### INTERNAL METHODS ####

    def _clean_trajectory_files(self):
        open(self.trajectory_path_json, 'w')
        open(self.trajectory_path_csv, 'w')

    def _build_pipeline_space(self):
        ps = PipelineSpace()
        tp = TestPreprocessingStep()
        tc = TestClassificationStep()
        ps.add_pipeline_steps([tp, tc])
        return ps


if __name__ == "__main__":
    d = Driver(data_path)
    d.run(caching=True, cache_directory=cache_directory, wallclock_limit=10, downsampling=2000, run_counter=1)
