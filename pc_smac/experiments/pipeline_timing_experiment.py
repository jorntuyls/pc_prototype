
import os

import argparse

from pc_smac.pc_smac.utils.data_loader import DataLoader
from pc_smac.pc_smac.utils.statistics import Statistics
from pc_smac.pc_smac.config_space.config_space_builder import ConfigSpaceBuilder
from pc_smac.pc_smac.pipeline_space.pipeline_space import PipelineSpace
from pc_smac.pc_smac.pipeline_space.pipeline_step import TestPreprocessingStep, TestClassificationStep
from pc_smac.pc_smac.pipeline.pipeline_runner import PipelineRunner, CachedPipelineRunner, PipelineTester
from pc_smac.pc_smac.pc_runhistory.pc_runhistory import PCRunHistory

from smac.smbo.objective import average_cost

def run_experiment(data_id, location, output_dir, nb_configs=100, seed=None, cache_directory=None, downsampling=None):

    # ouput directory
    if output_dir == None:
        output_dir = os.path.dirname(os.path.abspath(__file__)) + "/results/"

    # Build pipeline space
    pipeline_space = PipelineSpace()
    tp = TestPreprocessingStep()
    tc = TestClassificationStep()
    pipeline_space.add_pipeline_steps([tp, tc])

    # Build configuration space
    cs_builder = ConfigSpaceBuilder(pipeline_space)
    #print("SEED: {}".format(seed)) if seed else print("NOT SEED: {}".format(seed))
    config_space = cs_builder.build_config_space(seed=seed) if seed else cs_builder.build_config_space()

    config1 = config_space.sample_configuration(1)
    config2 = config_space.sample_configuration(1)
    #print(config1)
    #print(config2)

    # Sample configurations from configuration space
    rand_configs = config_space.sample_configuration(size=nb_configs) if nb_configs > 1 else [config_space.sample_configuration(size=nb_configs)]

    # Run the random configurations = pipelines on data set
    data_path = location + str(data_id) + "_bac" if location[-1] == "/" else location + "/" + str(data_id) + "_bac"
    data_set = data_path.split("/")[-1]
    output_dir = output_dir + data_set + "/" if output_dir[-1] == "/" else output_dir + "/" + data_set + "/"
    stamp = data_set + "_seed_" + str(seed)
    run_experiment_on_data(stamp=stamp,
                           data_path=data_path,
                           output_dir=output_dir,
                           pipeline_space=pipeline_space,
                           configs=rand_configs,
                           cache_directory=cache_directory,
                           downsampling=downsampling)

def run_experiment_on_data(stamp, data_path, output_dir, pipeline_space, configs, cache_directory, downsampling):
    # Make own output directory for each data set

    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load data
    data_loader = DataLoader(data_path)
    data = data_loader.get_data()

    for i in range(0, len(configs)):
        ## Run with caching disabled ##
        new_stamp = stamp + "_nocaching_config"
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
        #statistics.save()


        # ## Run with caching enabled first time ##
        # new_stamp = stamp + "_caching_config_run_1"
        # info = {
        #     'stamp': new_stamp,
        #     'caching': True,
        #     'cache_directory': cache_directory,
        #     'downsampling': downsampling
        # }
        # statistics = Statistics(new_stamp, output_dir,
        #                         information=info)
        # # Build runhistory
        # runhistory = PCRunHistory(average_cost)
        #
        # # Build pipeline runner
        # cached_pipeline_runner = CachedPipelineRunner(data, pipeline_space, runhistory, statistics,
        #                                        cache_directory=cache_directory,
        #                                        downsampling=downsampling)
        # # Start timer
        # statistics.start_timer()
        #
        # # Run pipeline
        # cached_pipeline_runner.start(config=configs[i])
        #
        # # Save statistics
        # #statistics.save()
        #
        #
        # ## Run with caching enabled second time ##
        # new_stamp = stamp + "_caching_config_run_2"
        # info = {
        #     'stamp': new_stamp,
        #     'caching': True,
        #     'cache_directory': cache_directory,
        #     'downsampling': downsampling
        # }
        # statistics = Statistics(new_stamp, output_dir,
        #                         information=info)
        #
        # # Give pipelinerunner new statistics object
        # cached_pipeline_runner.update_statistics(statistics)
        #
        # # Start timer
        # statistics.start_timer()
        #
        # # Run pipeline
        # cached_pipeline_runner.start(config=configs[i])
        #
        # # clean cache
        # cached_pipeline_runner.clean_cache()
        #
        # # Save statistics
        # #statistics.save()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nbconfigs", type=int, help="Number of configurations")
    parser.add_argument("-s", "--seed", type=int, help="Seed for sampling configurations")
    parser.add_argument("-d", "--dataid", type=int, help="Dataset id")
    parser.add_argument("-l", "--location", type=str, help="Dataset directory")
    parser.add_argument("-o", "--outputdir", type=str, default=None, help="Output directory")
    parser.add_argument("-ds", "--downsampling", type=int, default=None, help="Number of data points to downsample to")
    parser.add_argument("-cd", "--cachedir", type=str, default=None, help="Cache directory")
    args = parser.parse_args()

    return args




if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(args.dataid, args.location, args.outputdir, args.nbconfigs, args.seed, args.cachedir, args.downsampling)