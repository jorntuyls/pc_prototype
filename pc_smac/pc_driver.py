
__author__ = 'jorntuyls'

import os
import time

from smac.scenario.scenario import Scenario
from smac.smbo.objective import average_cost
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.utils.io.traj_logging import TrajLogger

from pc_smac.pc_smac.config_space.config_space_builder import ConfigSpaceBuilder
from pc_smac.pc_smac.data_loader.data_loader import DataLoader
from pc_smac.pc_smac.pc_runhistory.pc_runhistory import PCRunHistory
from pc_smac.pc_smac.pc_smbo.smbo_builder import SMBOBuilder
from pc_smac.pc_smac.pipeline.pipeline_runner import PipelineRunner, CachedPipelineRunner, PipelineTester
from pc_smac.pc_smac.pipeline_space.pipeline_space import PipelineSpace
from pc_smac.pc_smac.pipeline_space.pipeline_space_builder import PipelineSpaceBuilder
from pc_smac.pc_smac.pipeline_space.pipeline_step import OneHotEncodingStep, ImputationStep, RescalingStep, \
    BalancingStep, PreprocessingStep, ClassificationStep
from pc_smac.pc_smac.utils.statistics import Statistics


class Driver:

    def __init__(self, data_path, output_dir=None, pipeline_space_string=None):
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)

        # test
        #from autosklearn.pipeline.classification import SimpleClassificationPipeline
        #s = SimpleClassificationPipeline()
        #print(s.get_hyperparameter_search_space())
        self.pipeline_space = self._build_pipeline_space() if (pipeline_space_string == None) else self._parse_pipeline_space(pipeline_space_string)
        self.cs_builder = ConfigSpaceBuilder(self.pipeline_space)
        self.config_space = self.cs_builder.build_config_space()

        self.output_dir = output_dir if output_dir else os.path.dirname(os.path.abspath(__file__)) + "/output/"
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        except FileExistsError:
            pass

    def initialize(self, stamp, acq_func, random_leaf_size, cache_directory, wallclock_limit, runcount_limit, cutoff, memory_limit, downsampling, intensification_fold_size):
        # Check if caching is enabled
        caching = True if acq_func[:2] == "pc" else False

        # Check if cache_directory exists
        try:
            if cache_directory and not os.path.exists(cache_directory):
                os.makedirs(cache_directory)
        except FileExistsError:
            pass

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
            pr = CachedPipelineRunner(self.data, self.data_loader.info, self.pipeline_space, runhistory,
                                                   self.statistics,
                                                   cache_directory=cache_directory,
                                                   downsampling=downsampling,
                                                    num_cross_validation_folds=intensification_fold_size)
        else:
            pr = PipelineRunner(self.data, self.data_loader.info, self.pipeline_space, runhistory, self.statistics,
                                             downsampling=downsampling,
                                            num_cross_validation_folds=intensification_fold_size)

        # Choose acquisition function
        if acq_func in ["eips", "m-eips", "pc-m-eips", "pceips", "pc-m-pceips"]:
            model_target_names = ['cost', 'time']
        elif acq_func in ["ei", "m-ei", "pc-m-ei"]:
            model_target_names = ['cost']
        else:
            # Not a valid acquisition function
            raise ValueError("The provided acquisition function is not valid")

        trajectory_path = self.output_dir + "/logging/" + stamp # + self.data_path.split("/")[-1] + "/" + str(stamp)
        if not os.path.exists(trajectory_path):
            os.makedirs(trajectory_path)
        self.trajectory_path_json = trajectory_path + "/traj_aclib2.json"
        self.trajectory_path_csv = trajectory_path + "/traj_old.csv"

        # Build scenario
        intensification_instances = [[1]] if intensification_fold_size == None else [[i] for i in range(0, intensification_fold_size)]
        args = {'cs': self.config_space,
                'run_obj': "quality",
                'runcount_limit': runcount_limit,
                'wallclock_limit': wallclock_limit,
                'memory_limit': memory_limit,
                'cutoff_time': cutoff,
                'deterministic': "true",
                'abort_on_first_run_crash': "false",
                'instances': intensification_instances
                }
        scenario = Scenario(args)

        # Build stats
        stats = Stats(scenario,
                      output_dir=self.output_dir + "/smac/",
                      stamp=stamp)

        # Build tae runner
        tae_runner = ExecuteTAFuncDict(ta=pr.run,
                                       stats=stats,
                                       runhistory=runhistory,
                                       run_obj=scenario.run_obj,
                                       memory_limit=scenario.memory_limit)

        # Build SMBO object
        intensification_instances = [1] if intensification_fold_size == None else [i for i in range(0, intensification_fold_size)]
        smbo_builder = SMBOBuilder()
        self.smbo = smbo_builder.build_pc_smbo(
            tae_runner=tae_runner,
            stats=stats,
            scenario=scenario,
            runhistory=runhistory,
            aggregate_func=average_cost,
            acq_func_name=acq_func,
            model_target_names=model_target_names,
            logging_directory=trajectory_path,
            random_leaf_size=random_leaf_size,
            constant_pipeline_steps=["one_hot_encoder", "imputation", "rescaling",
                                     "balancing", "feature_preprocessor"],
            variable_pipeline_steps=["classifier"],
            intensification_instances=intensification_instances)


    def run(self,
            stamp=time.time(),
            acq_func="ei",
            random_leaf_size=1,
            wallclock_limit=3600,
            runcount_limit=10000,
            memory_limit=6000,
            cutoff=3600,
            cache_directory=None,
            downsampling=None,
            intensification_fold_size=None):

        # Initialize SMBO
        self.initialize(stamp=stamp,
                        acq_func=acq_func,
                        random_leaf_size=random_leaf_size,
                        cache_directory=cache_directory,
                        wallclock_limit=wallclock_limit,
                        runcount_limit=runcount_limit,
                        memory_limit=memory_limit,
                        cutoff=cutoff,
                        downsampling=downsampling,
                        intensification_fold_size=intensification_fold_size)

        # clean trajectory files
        self._clean_trajectory_files()

        # Start timer and clean statistics files
        self.statistics.start_timer()
        self.statistics.clean_files()

        # Run SMBO
        incumbent = self.smbo.run()

        # Save statistics
        # self.statistics.save()

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
        pt = PipelineTester(self.data, self.data_loader.info, self.pipeline_space, downsampling=downsampling)

        for traj in trajectory:
            traj['test_performance'] = pt.get_error(traj['incumbent'])

        return trajectory

    #### INTERNAL METHODS ####

    def _clean_trajectory_files(self):
        open(self.trajectory_path_json, 'w')
        open(self.trajectory_path_csv, 'w')

    def _build_pipeline_space(self):
        ps = PipelineSpace()
        o_s = OneHotEncodingStep()
        i_s = ImputationStep()
        r_s = RescalingStep()
        b_s = BalancingStep()
        p_s = PreprocessingStep()
        c_s = ClassificationStep()
        ps.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s]) #[p_s, c_s])
        return ps

    def _parse_pipeline_space(self, pipeline_space_string):
        preprocessor_names = pipeline_space_string.split("-")[0].split(",")
        classifier_names = pipeline_space_string.split("-")[1].split(",")

        pipeline_space_builder = PipelineSpaceBuilder()

        return pipeline_space_builder.build_pipeline_space(preprocessor_names=preprocessor_names,
                                                           classifier_names=classifier_names)



# if __name__ == "__main__":
#     from pc_smac.pc_smac.local_data_paths import data_path, output_dir, cache_directory
#     d = Driver(data_path=data_path, output_dir=output_dir)
#     d.run(stamp="test",
#          acq_func="pceips",
#          wallclock_limit=10,
#          runcount_limit=2,
#          memory_limit=4000000,
#          cutoff=3600,
#          cache_directory=cache_directory,
#          downsampling=1000)
