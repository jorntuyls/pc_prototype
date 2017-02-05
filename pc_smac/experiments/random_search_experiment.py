import os
import argparse

from pc_smac.pc_smac.utils.data_loader import DataLoader
from pc_smac.pc_smac.utils.statistics import Statistics
from pc_smac.pc_smac.config_space.config_space_builder import ConfigSpaceBuilder
from pc_smac.pc_smac.pipeline_space.pipeline_space import PipelineSpace
from pc_smac.pc_smac.pipeline_space.pipeline_step import PipelineStep, OneHotEncodingStep, ImputationStep, RescalingStep, \
    BalancingStep, PreprocessingStep, ClassificationStep
from pc_smac.pc_smac.random_search.random_search import RandomSearch, TreeRandomSearch
from pc_smac.pc_smac.pipeline.pipeline_runner import PipelineRunner, CachedPipelineRunner, PipelineTester

from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.kernel_pca import KernelPcaNode

from pc_smac.pc_smac.pipeline_space.classification_nodes.sgd import SGDNode



def run_random_search(stamp, data_path, version, wallclock_limit, seed=None, output_dir=None, cache_directory=None, downsampling=None):
    # data set
    data_set = data_path.split("/")[-1]

    # cache directory
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)

    # ouput directory
    if output_dir == None:
        output_dir = os.path.dirname(os.path.abspath(__file__)) + "/results/"
    else:
        output_dir = output_dir + data_set + "/"  if output_dir[-1] == "/" \
                else output_dir + "/" + data_set + "/"

    # load data
    data_loader = DataLoader(data_path)
    data = data_loader.get_data()

    # Build pipeline space
    pipeline_space = PipelineSpace()
    o_s = OneHotEncodingStep()
    i_s = ImputationStep()
    r_s = RescalingStep()
    b_s = BalancingStep()
    p_s = PipelineStep(name='feature_preprocessor', nodes=[KernelPcaNode()])
    c_s = PipelineStep(name='classifier', nodes=[SGDNode()])
    pipeline_space.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s])

    # Build configuration space
    cs_builder = ConfigSpaceBuilder(pipeline_space)
    config_space = cs_builder.build_config_space(seed=seed)

    # Build statistics
    info = {
        'data_location': data_path,
        'stamp': stamp,
        'version': version,
        'wallclock_limit': wallclock_limit,
        'seed': seed,
        'downsampling': downsampling
    }
    statistics = Statistics(stamp,
                            output_dir,
                            information=info,
                            total_runtime=wallclock_limit)

    # Build pipeline runner
    if version == 'tree':
        pipeline_runner = CachedPipelineRunner(data=data,
                                               pipeline_space=pipeline_space,
                                               runhistory=None,
                                               statistics=statistics,
                                               cache_directory=cache_directory,
                                               downsampling=downsampling)
        random_search = TreeRandomSearch(config_space=config_space,
                                         pipeline_runner=pipeline_runner,
                                         wallclock_limit=wallclock_limit,
                                         statistics=statistics,
                                         constant_pipeline_steps=["one_hot_encoder", "imputation", "rescaling",
                                                                  "balancing", "feature_preprocessor"],
                                         variable_pipeline_steps=["classifier"],
                                         number_leafs_split=4)
    else:
        pipeline_runner = PipelineRunner(data=data,
                                         pipeline_space=pipeline_space,
                                         runhistory=None,
                                         statistics=statistics,
                                         downsampling=downsampling)
        random_search = RandomSearch(config_space=config_space,
                                     pipeline_runner=pipeline_runner,
                                     wallclock_limit=wallclock_limit,
                                     statistics=statistics)

    # Run random search
    print("start random search")
    incumbent = random_search.run()
    print("... end random search")

    # test performance of incumbents
    incumbent_trajectory = statistics.get_incumbent_trajectory(config_space=config_space)
    trajectory = run_tests(data, incumbent_trajectory, pipeline_space, downsampling=downsampling)
    print(trajectory)

    # Save new trajectory to output directory
    # First transform the configuration to a dictionary
    for traj in trajectory:
        traj['incumbent'] = traj['incumbent'].get_dictionary()
    statistics.add_incumbents_trajectory(trajectory)

    return incumbent


def run_tests(data, trajectory, pipeline_space, downsampling=None):
    pt = PipelineTester(data, pipeline_space, downsampling=downsampling)

    for traj in trajectory:
        traj['test_performance'] = pt.get_performance(traj['incumbent'])

    return trajectory


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=str, help="Random search version")
    parser.add_argument("-w", "--wallclock", type=int, help="Wallclock limit")
    parser.add_argument("-l", "--location", type=str, help="Data location")
    parser.add_argument("-st", "--stamp", type=str, default="stamp", help="Stamp for output files")
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
                      seed=args.seed,
                      output_dir=args.outputdir,
                      cache_directory=args.cachedir,
                      downsampling=args.downsampling)