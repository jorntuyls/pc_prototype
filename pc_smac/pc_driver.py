
__author__ = 'jorntuyls'

import os
import time

from pc_smac.pc_smac.utils.data_loader import DataLoader
from pc_smac.pc_smac.config_space.config_space_builder import ConfigSpaceBuilder
from pc_smac.pc_smac.pipeline_space.pipeline_space import PipelineSpace
from pc_smac.pc_smac.pipeline_space.pipeline_step import TestPreprocessingStep, TestClassificationStep
from pc_smac.pc_smac.pipeline.pipeline_runner import PipelineRunner, CachedPipelineRunner, PipelineTester
from pc_smac.pc_smac.pc_smbo.smbo_builder import SMBOBuilder
from pc_smac.pc_smac.pc_runhistory.pc_runhistory import PCRunHistory
from pc_smac.pc_smac.utils.io_utils import save_trajectory_for_plotting
from pc_smac.pc_smac.utils.statistics import Statistics
from smac.scenario.scenario import Scenario

from smac.smbo.objective import average_cost
from smac.utils.io.traj_logging import TrajLogger
from smac.tae.execute_func import ExecuteTAFuncDict

class Driver:

    def __init__(self, data_path, output_dir=None):
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)

        self.pipeline_space = self._build_pipeline_space()

        self.cs_builder = ConfigSpaceBuilder(self.pipeline_space)
        self.config_space = self.cs_builder.build_config_space()

        self.output_dir = output_dir if output_dir else os.path.dirname(os.path.abspath(__file__)) + "/output/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initialize(self, stamp, acq_func, cache_directory, wallclock_limit, runcount_limit, cutoff, memory_limit, downsampling):
        # Check if caching is enabled
        caching = True if acq_func[:2] == "pc" else False

        # Load data
        self.data = self.data_loader.get_data()

        # Build runhistory
        # TODO Does this work correctly for non-caching?
        runhistory = PCRunHistory(average_cost)

        # Setup statistics
        info = {
            'stamp': stamp,
            'caching': caching,
            'acquisition_function': acq_func,
            'cache_directory': cache_directory,
            'wallclock_limit': wallclock_limit,
            'downsampling': downsampling
        }

        self.statistics = Statistics(stamp,
                                     self.output_dir,
                                     information=info,
                                     total_runtime=wallclock_limit)

        # Set cache directory
        if caching:
            # Check if directory exists, otherwise create it
            if cache_directory != None and not os.path.exists(cache_directory):
                os.makedirs(cache_directory)
            pr = CachedPipelineRunner(self.data, self.pipeline_space, runhistory,
                                                   self.statistics,
                                                   cache_directory=cache_directory,
                                                   downsampling=downsampling)
        else:
            pr = PipelineRunner(self.data, self.pipeline_space, runhistory, self.statistics,
                                             downsampling=downsampling)

        # Choose acquisition function
        if acq_func == "pceips" or acq_func == "eips":
            model_target_names = ['cost', 'time']
        elif acq_func == "ei":
            model_target_names = ['cost']
        else:
            # Not a valid acquisition function
            raise ValueError("The provided acquisition function is not valid")

        trajectory_path = os.path.dirname(os.path.abspath(__file__)) + "/logging/" + self.data_path.split("/")[-1] + "/" + str(stamp)
        if not os.path.exists(trajectory_path):
            os.makedirs(trajectory_path)
        self.trajectory_path_json = trajectory_path + "/traj_aclib2.json"
        self.trajectory_path_csv = trajectory_path + "/traj_old.csv"

        # Build scenario
        args = {'cs': self.config_space,
                'run_obj': "quality",
                'runcount_limit': runcount_limit,
                'wallclock_limit': wallclock_limit,
                'memory_limit': memory_limit,
                'cutoff_time': cutoff,
                'deterministic': "true"
                }
        scenario = Scenario(args)

        # Build SMBO object
        smbo_builder = SMBOBuilder()
        self.smbo = smbo_builder.build_pc_smbo(
            execute_func=pr.run,
            scenario=scenario,
            runhistory=runhistory,
            aggregate_func=average_cost,
            acq_func_name=acq_func,
            model_target_names=model_target_names,
            logging_directory=trajectory_path)


    def run(self,
            stamp=time.time(),
            acq_func="ei",
            wallclock_limit=3600,
            runcount_limit=10000,
            memory_limit=6000,
            cutoff=3600,
            cache_directory=None,
            downsampling=None):

        # Initialize SMBO
        self.initialize(stamp=stamp,
                        acq_func=acq_func,
                        cache_directory=cache_directory,
                        wallclock_limit=wallclock_limit,
                        runcount_limit=runcount_limit,
                        memory_limit=memory_limit,
                        cutoff=cutoff,
                        downsampling=downsampling)

        # clean trajectory files
        self._clean_trajectory_files()

        # Start timer and clean statistics files
        self.statistics.start_timer()
        self.statistics.clean_files()

        # Run SMBO
        incumbent = self.smbo.run()

        # Save statistics
        self.statistics.save()

        # Read trajectory files with incumbents and retrieve test performances
        self.trajectory = TrajLogger.read_traj_aclib_format(self.trajectory_path_json, self.config_space)
        trajectory = self.run_tests(self.trajectory, downsampling=downsampling)

        # Save new trajectory to output directory
        # First transform the configuration to a dictionary
        for traj in trajectory:
            traj['incumbent'] = traj['incumbent'].get_dictionary()
        self.statistics.add_incumbents_trajectory(trajectory)

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
    from pc_smac.pc_smac.local_data_paths import data_path, output_dir, cache_directory
    d = Driver(data_path=data_path, output_dir=output_dir)
    d.run(stamp="test",
         acq_func="pceips",
         wallclock_limit=10,
         runcount_limit=2,
         memory_limit=4000000,
         cutoff=3600,
         cache_directory=cache_directory,
         downsampling=1000)
