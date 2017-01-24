
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

from smac.smbo.objective import average_cost
from smac.utils.io.traj_logging import TrajLogger

class Driver:

    def __init__(self, data_path, output_dir=None):
        self.data_loader = DataLoader(data_path)

        self.pipeline_space = self._build_pipeline_space()

        self.cs_builder = ConfigSpaceBuilder(self.pipeline_space)
        self.config_space = self.cs_builder.build_config_space()

        self.output_dir = output_dir if output_dir else os.path.dirname(os.path.abspath(__file__)) + "/output/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # TODO make names not hardcoded
        trajectory_path = os.path.dirname(os.path.abspath(__file__)) + "/logging"
        if not os.path.exists(trajectory_path):
            os.makedirs(trajectory_path)
        self.trajectory_path_json = trajectory_path + "/traj_aclib2.json"
        self.trajectory_path_csv = trajectory_path + "/traj_old.csv"

    def initialize(self, stamp, acq_func, cache_directory, wallclock_limit, downsampling):
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
        self.statistics = Statistics(stamp, self.output_dir,
                                information=info,
                                total_runtime=wallclock_limit)

        # Set cache directory
        if caching:
            # Check if directory exists, otherwise create it
            if cache_directory != None and not os.path.exists(cache_directory):
                os.makedirs(cache_directory)
            self.tae_runner = CachedPipelineRunner(self.data, self.pipeline_space, runhistory,
                                                   self.statistics,
                                                   cache_directory=cache_directory,
                                                   downsampling=downsampling)
        else:
            self.tae_runner = PipelineRunner(self.data, self.pipeline_space, runhistory, self.statistics,
                                             downsampling=downsampling)

        # Choose acquisition function
        if acq_func == "pceips" or acq_func == "eips":
            model_target_names = ['cost', 'time']
        elif acq_func == "ei":
            model_target_names = ['cost']
        else:
            # Not a valid acquisition function
            raise ValueError("The provided acquisition function is not valid")

        # Build SMBO object
        smbo_builder = SMBOBuilder()
        self.smbo = smbo_builder.build_pc_smbo(
            config_space=self.config_space,
            tae_runner=self.tae_runner,
            runhistory=runhistory,
            aggregate_func=average_cost,
            acq_func_name=acq_func,
            model_target_names=model_target_names,
            logging_directory=os.path.dirname(os.path.abspath(__file__)) + "/logging",
            wallclock_limit=wallclock_limit)


    def run(self,
            stamp=time.time(),
            acq_func="ei",
            cache_directory=None,
            wallclock_limit=3600,
            downsampling=None):
        # clean trajectory files
        self._clean_trajectory_files()

        # Initialize SMBO
        self.initialize(stamp=stamp,
                        acq_func=acq_func,
                        cache_directory=cache_directory,
                        wallclock_limit=wallclock_limit,
                        downsampling=downsampling)

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

