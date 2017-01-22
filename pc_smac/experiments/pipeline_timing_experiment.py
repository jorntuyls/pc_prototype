
import os

from pc_prototype.pc_smac.utils.data_loader import DataLoader
from pc_prototype.pc_smac.utils.statistics import Statistics
from pc_prototype.pc_smac.config_space.config_space_builder import ConfigSpaceBuilder
from pc_prototype.pc_smac.pipeline_space.pipeline_space import PipelineSpace
from pc_prototype.pc_smac.pipeline_space.pipeline_step import TestPreprocessingStep, TestClassificationStep
from pc_prototype.pc_smac.pipeline.pipeline_runner import PipelineRunner, CachedPipelineRunner, PipelineTester
from pc_prototype.pc_smac.pc_runhistory.pc_runhistory import PCRunHistory

from smac.smbo.objective import average_cost

from pc_prototype.pc_smac.data_paths import all_data_paths, cache_directory

def run_experiment(nb_configs=100, downsampling=None):

    # ouput directory
    output_dir = os.path.dirname(os.path.abspath(__file__)) + "/results/"

    # Build pipeline space
    pipeline_space = PipelineSpace()
    tp = TestPreprocessingStep()
    tc = TestClassificationStep()
    pipeline_space.add_pipeline_steps([tp, tc])

    # Build configuration space
    cs_builder = ConfigSpaceBuilder(pipeline_space)
    config_space = cs_builder.build_config_space()

    # Sample configurations from configuration space
    rand_configs = config_space.sample_configuration(size=nb_configs)

    # Run the random configurations = pipelines on each data set
    for data_path in all_data_paths:
        stamp = data_path.split("/")[-1]
        run_experiment_on_data(stamp=stamp,
                               data_path=data_path,
                               output_dir=output_dir,
                               pipeline_space=pipeline_space,
                               configs=rand_configs,
                               downsampling=downsampling)

def run_experiment_on_data(stamp, data_path, output_dir, pipeline_space, configs, downsampling):
    # Make own output directory for each data set
    output_dir = output_dir + stamp + "/"
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load data
    data_loader = DataLoader(data_path)
    data = data_loader.get_data()

    for i in range(0, len(configs)):
        ## Run with caching disabled ##
        new_stamp = stamp + "_nocaching_config_" + str(i+1)
        info = {
            'stamp': new_stamp,
            'caching': False,
            'downsampling': downsampling
        }
        statistics = Statistics(new_stamp, output_dir,
                                     information=info)
        # Build runhistory
        runhistory = PCRunHistory(average_cost)

        # Build pipeline runner
        pipeline_runner = PipelineRunner(data, pipeline_space, runhistory, statistics,
                       downsampling=downsampling)
        # Start timer
        statistics.start_timer()

        # Run pipeline
        pipeline_runner.start(config=configs[i])

        # Save statistics
        statistics.save()


        ## Run with caching enabled first time ##
        new_stamp = stamp + "_caching_config_" + str(i+1) + "_run_1"
        info = {
            'stamp': new_stamp,
            'caching': True,
            'cache_directory': cache_directory,
            'downsampling': downsampling
        }
        statistics = Statistics(new_stamp, output_dir,
                                information=info)
        # Build runhistory
        runhistory = PCRunHistory(average_cost)

        # Build pipeline runner
        cached_pipeline_runner = CachedPipelineRunner(data, pipeline_space, runhistory, statistics,
                                               cache_directory=cache_directory,
                                               downsampling=downsampling)
        # Start timer
        statistics.start_timer()

        # Run pipeline
        cached_pipeline_runner.start(config=configs[i])

        # Save statistics
        statistics.save()


        ## Run with caching enabled second time ##
        new_stamp = stamp + "_caching_config_" + str(i+1) + "_run_2"
        info = {
            'stamp': new_stamp,
            'caching': True,
            'cache_directory': cache_directory,
            'downsampling': downsampling
        }
        statistics = Statistics(new_stamp, output_dir,
                                information=info)

        # Give pipelinerunner new statistics object
        cached_pipeline_runner.update_statistics(statistics)

        # Start timer
        statistics.start_timer()

        # Run pipeline
        cached_pipeline_runner.start(config=configs[i])

        # Save statistics
        statistics.save()




if __name__ == "__main__":
    run_experiment(nb_configs=2, downsampling=2000)