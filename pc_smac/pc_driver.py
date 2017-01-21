
__author__ = 'jorntuyls'

import os


from utils.data_loader import DataLoader
from config_space.config_space_builder import ConfigSpaceBuilder
from pipeline_space.pipeline_space import PipelineSpace
from pipeline_space.pipeline_step import TestPreprocessingStep, TestClassificationStep
from pipeline.pipeline_runner import PipelineRunner, CachedPipelineRunner, PipelineTester
from pc_smbo.smbo_builder import SMBOBuilder
from pc_runhistory.pc_runhistory import PCRunHistory
from utils.io_utils import save_trajectory_to_plotting_format

from smac.smbo.objective import average_cost
from smac.utils.io.traj_logging import TrajLogger

from data_paths import data_path, cache_directory

class Driver:

    def __init__(self, data_path):
        self.data_loader = DataLoader(data_path)

        self.pipeline_space = self._build_pipeline_space()

        self.cs_builder = ConfigSpaceBuilder(self.pipeline_space)
        self.config_space = self.cs_builder.build_config_space()

        # TODO make names not hardcoded
        self.trajectory_path_json = os.path.dirname(os.path.abspath(__file__)) + "/logging/traj_aclib2.json"
        self.trajectory_path_csv = os.path.dirname(os.path.abspath(__file__)) + "/logging/traj_old.csv"

    def initialize(self,
                   caching=True,
                   cache_directory=None,
                   wallclock_limit=3600,
                   downsampling=None):
        # Load data
        self.data = self.data_loader.get_data()

        # Build runhistory
        # TODO Does this work correctly for non-caching?
        runhistory = PCRunHistory(average_cost)

        # Set cache directory
        if caching:
            cache_dir = cache_directory
            self.tae_runner = CachedPipelineRunner(self.data, self.pipeline_space, runhistory,
                                                   cache_directory=cache_dir,
                                                   downsampling=downsampling)
        else:
            self.tae_runner = PipelineRunner(self.data, self.pipeline_space, runhistory,
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
            caching=True,
            cache_directory=None,
            wallclock_limit=3600,
            downsampling=None,
            run_counter=1,
            plot_time=10):
        # clean trajectory files
        self._clean_trajectory_files()

        # Initialize SMBO
        self.initialize(caching=caching, cache_directory=cache_directory,
                        wallclock_limit=wallclock_limit, downsampling=downsampling)

        # Run SMBO
        incumbent = self.smbo.run()

        # Read trajectory files with incumbents and retrieve test performances
        self.trajectory = TrajLogger.read_traj_aclib_format(self.trajectory_path_json, self.config_space)
        self.run_tests(self.trajectory, downsampling=downsampling)
        #print(self.trajectory)
        # Save new trajectory for plotting
        self._save_trajectory_for_plotting(self.trajectory,
                                           wallclock_limit=wallclock_limit,
                                           plot_time=plot_time,
                                           caching=caching,
                                           run_counter=run_counter)

        return incumbent


    def run_tests(self, trajectory, downsampling=None):
        pt = PipelineTester(self.data, self.pipeline_space, downsampling=downsampling)

        for traj in trajectory:
            traj['test_performance'] = pt.get_performance(traj['incumbent'])

        return trajectory

    # Internal methods

    def _save_trajectory_for_plotting(self, trajectory, wallclock_limit, plot_time, caching, run_counter=1):
        filename = "validationResults-traj-caching=" + str(caching) + "-run=" + str(run_counter) + ".csv"
        destination = os.path.dirname(os.path.abspath(__file__)) + "/results"
        # Create new trajectory file with choosen timestamps
        traj_0 = trajectory[0]
        new_trajectory = [{
            'wallclock_time': 0.0,
            'training_performance': traj_0['cost'],
            'test_performance': traj_0['test_performance']
        }]

        time = plot_time
        while time <= wallclock_limit:
            t = None
            # find trajectory element that is the incumbent at time 'time'
            for i in range(0,len(trajectory)-1):
                traj_i = trajectory[i]
                traj_i1 = trajectory[i+1]
                if time >= traj_i['wallclock_time'] and time < traj_i1['wallclock_time'] :
                    t = traj_i
                    break
                elif time > trajectory[-1]['wallclock_time']\
                        :
                    t = trajectory[-1]
            if t:
                new_trajectory.append({
                    'wallclock_time': time,
                    'training_performance': t['cost'],
                    'test_performance': t['test_performance']
                })
            else:
                raise ValueError("trajectory element should never be none")

            time += plot_time
        print(new_trajectory)
        save_trajectory_to_plotting_format(new_trajectory, destination=destination, filename=filename)



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
