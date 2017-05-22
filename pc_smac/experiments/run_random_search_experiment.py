import argparse
import os

from pc_smac.pc_smac.config_space.config_space_builder import ConfigSpaceBuilder
from pc_smac.pc_smac.data_loader.data_loader import DataLoader
from pc_smac.pc_smac.pipeline.pipeline_runner import PipelineRunner, CachedPipelineRunner, PipelineTester
from pc_smac.pc_smac.pipeline_space.pipeline_space import PipelineSpace
from pc_smac.pc_smac.pipeline_space.pipeline_step import OneHotEncodingStep, ImputationStep, RescalingStep, \
    BalancingStep, PreprocessingStep, ClassificationStep
from pc_smac.pc_smac.random_search.random_search import RandomSearch, TreeRandomSearch
from pc_smac.pc_smac.utils.statistics import Statistics


def run_random_search(stamp, data_path, version, wallclock_limit, run_limit, memory_limit, cutoff, splitting_number, random_splitting_enabled,
                      seed=None, output_dir=None, cache_directory=None, downsampling=None):
    # data set
    data_set = data_path.split("/")[-1]

    # cache directory
    try:
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)
    except FileExistsError:
        pass

    # ouput directory
    try:
        if output_dir == None:
            output_dir = os.path.dirname(os.path.abspath(__file__)) + "/results/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except FileExistsError:
        pass

    # load data
    data_loader = DataLoader(data_path)
    data = data_loader.get_data()
    dataset_properties = data_loader.info

    # Build pipeline space
    pipeline_space = PipelineSpace()
    o_s = OneHotEncodingStep()
    i_s = ImputationStep()
    r_s = RescalingStep()
    b_s = BalancingStep()
    p_s = PreprocessingStep() # PipelineStep(name='feature_preprocessor', nodes=[KernelPcaNode()])
    c_s = ClassificationStep() # PipelineStep(name='classifier', nodes=[SGDNode()])
    pipeline_space.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s])

    # Build configuration space
    cs_builder = ConfigSpaceBuilder(pipeline_space)
    config_space = cs_builder.build_config_space(seed=seed, dataset_properties=dataset_properties)

    # Build statistics
    info = {
        'data_location': data_path,
        'stamp': stamp,
        'version': version,
        'wallclock_limit': wallclock_limit,
        'memory_limit': memory_limit,
        'cutoff': cutoff,
        'seed': seed,
        'downsampling': downsampling
    }
    statistics = Statistics(stamp,
                            output_dir,
                            information=info,
                            total_runtime=wallclock_limit,
                            run_limit=run_limit)
    statistics.clean_files()

    # The pipeline parts that can get cached
    cached_pipeline_steps = [["one_hot_encoder", "imputation", "rescaling",
                                                "balancing", "feature_preprocessor"]]

    num_cross_validation_folds = 10
    # Build pipeline runner
    if version == '2step':
        pipeline_runner = CachedPipelineRunner(data=data,
                                               data_info=dataset_properties,
                                               pipeline_space=pipeline_space,
                                               runhistory=None,
                                               cached_pipeline_steps=cached_pipeline_steps,
                                               statistics=statistics,
                                               cache_directory=cache_directory,
                                               downsampling=downsampling,
                                               num_cross_validation_folds=num_cross_validation_folds)
        random_search = TreeRandomSearch(config_space=config_space,
                                         pipeline_runner=pipeline_runner,
                                         wallclock_limit=wallclock_limit,
                                         memory_limit=memory_limit,
                                         statistics=statistics,
                                         constant_pipeline_steps=["one_hot_encoder", "imputation", "rescaling",
                                                                  "balancing", "feature_preprocessor"],
                                         variable_pipeline_steps=["classifier"],
                                         splitting_number=splitting_number,
                                         random_splitting_enabled=random_splitting_enabled)
    else:
        pipeline_runner = PipelineRunner(data=data,
                                         data_info=dataset_properties,
                                         pipeline_space=pipeline_space,
                                         runhistory=None,
                                         statistics=statistics,
                                         downsampling=downsampling,
                                         num_cross_validation_folds=num_cross_validation_folds)
        random_search = RandomSearch(config_space=config_space,
                                     pipeline_runner=pipeline_runner,
                                     wallclock_limit=wallclock_limit,
                                     memory_limit=memory_limit,
                                     statistics=statistics)

    # Run random search
    print("start random search")
    incumbent = random_search.run(cutoff=cutoff)
    print("... end random search")

    # test performance of incumbents
    incumbent_trajectory = statistics.get_incumbent_trajectory(config_space=config_space)
    trajectory = run_tests(data, dataset_properties, incumbent_trajectory, pipeline_space, downsampling=downsampling)
    print(trajectory)

    # Save new trajectory to output directory
    # First transform the configuration to a dictionary
    for traj in trajectory:
        traj['incumbent'] = traj['incumbent'].get_dictionary()
    statistics.add_incumbents_trajectory(trajectory)

    return incumbent


def run_tests(data, dataset_properties, trajectory, pipeline_space, downsampling=None):
    pt = PipelineTester(data, dataset_properties, pipeline_space, downsampling=downsampling)

    for traj in trajectory:
        traj['test_performance'] = pt.get_error(traj['incumbent'])

    return trajectory


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=str, help="Random search version")
    parser.add_argument("-w", "--wallclock", type=int, help="Wallclock limit")
    parser.add_argument("-r", "--run_limit", type=int, default=10000, help="Run limit")
    parser.add_argument("-m", "--memory", type=int, help="Memory limit")
    parser.add_argument("-c", "--cutoff", type=int, help="Cutoff")
    parser.add_argument("-l", "--location", type=str, help="Data location")
    parser.add_argument("-st", "--stamp", type=str, default="stamp", help="Stamp for output files")
    parser.add_argument("-sn", "--splitting_number", type=int, default=10, help="Splitting number")
    parser.add_argument("-rs", "--random_splitting", type=bool, default=False, help="Downsampling of data")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed for random")
    parser.add_argument("-o", "--outputdir", type=str, default=None, help="Output directory")
    parser.add_argument("-cd", "--cachedir", type=str, default=None, help="Cache directory")
    parser.add_argument("-ds", "--downsampling", type=int, default=None, help="Downsampling of data")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_random_search(stamp=args.stamp,
                      data_path=args.location,
                      version=args.version,
                      wallclock_limit=args.wallclock,
                      run_limit=args.run_limit,
                      memory_limit=args.memory,
                      cutoff=args.cutoff,
                      splitting_number=args.splitting_number,
                      random_splitting_enabled=args.random_splitting,
                      seed=args.seed,
                      output_dir=args.outputdir,
                      cache_directory=args.cachedir,
                      downsampling=args.downsampling)