import os
import argparse

from pc_smac.pc_smac.whitebox_driver import WhiteboxDriverRandomSearch
from pc_smac.pc_smac.whitebox.paraboloid import Paraboloid, CachedParaboloid, CachedParaboloid2Minima, Paraboloid2Minima
from pc_smac.pc_smac.whitebox.beale_function import Beale, CachedBeale
from pc_smac.pc_smac.utils.statistics import WhiteboxStatistics

def run_whitebox_experiment_random_search(number_of_leafs,
                                          wallclock_limit,
                                          seed=None,
                                          nb_experiments=20,
                                          test_function = "paraboloid",
                                          min_x = [0.90, 0.90],
                                          min_y = [0.1, 0.9],
                                          output_dir=None):
    output_dir_pattern = output_dir + "{}" if output_dir[-1] == '/' else output_dir + "/{}"
    versions = ['default', '2step']
    for version in versions:
        new_output_dir = output_dir_pattern.format(version)
        for i in range(0, nb_experiments):
            wd = WhiteboxDriverRandomSearch(output_dir=new_output_dir)
            wd.run(stamp=version + "_run_" + str(i),
                  version=version,
                  number_of_leafs=number_of_leafs,
                  wallclock_limit=wallclock_limit,
                  seed=seed,
                  test_function=test_function,
                  min_x=min_x,
                  min_y=min_y)



def run_random_search(stamp,
                      version,
                      number_of_leafs,
                      wallclock_limit,
                      seed=None,
                      test_function = "paraboloid",
                      min_x = [0.90, 0.90],
                      min_y = [0.1, 0.9],
                      output_dir=None):

    # ouput directory
    if output_dir == None:
        output_dir = os.path.dirname(os.path.abspath(__file__)) + "/results/"

    # Build statistics
    info = {
        'version': version,
        'wallclock_limit': wallclock_limit,
        'seed': seed,
    }
    statistics = WhiteboxStatistics(stamp,
                                     output_dir,
                                     information=info,
                                     total_runtime=wallclock_limit)

    statistics.clean_files()

    if version == '2step':
        if test_function == "beale":
            tae = CachedBeale(runhistory=None,
                              statistics=statistics,
                              min_x=min_x,
                              min_y=min_y)
        else:
            tae = CachedParaboloid2Minima(runhistory=None,
                                          statistics=statistics,
                                          min_x=min_x,
                                          min_y=min_y)
        # Build configuration space
        config_space = tae.get_config_space(seed=seed)
        random_search = Whitebox2stepRandomSearch(config_space=config_space,
                                         pipeline_runner=tae,
                                         wallclock_limit=wallclock_limit,
                                         statistics=statistics,
                                         constant_pipeline_steps=["preprocessor"],
                                         variable_pipeline_steps=["classifier"],
                                         number_leafs_split=number_of_leafs)
    else:
        if test_function == "beale":
            tae = Beale(runhistory=None,
                        statistics=statistics,
                        min_x=min_x,
                        min_y=min_y)
        else:
            tae = Paraboloid2Minima(runhistory=None,
                                    statistics=statistics,
                                    min_x=min_x,
                                    min_y=min_y)
        # Build configuration space
        config_space = tae.get_config_space(seed=seed)
        random_search = WhiteboxRandomSearch(config_space=config_space,
                                     pipeline_runner=tae,
                                     wallclock_limit=wallclock_limit,
                                     statistics=statistics)

    # Run random search
    print("start random search")
    incumbent = random_search.run()
    print("... end random search")

    trajectory = statistics.get_incumbent_trajectory_dicts()
    statistics.add_incumbents_trajectory(trajectory=trajectory)

    return incumbent


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wallclock", type=int, help="Wallclock limit")
    parser.add_argument("-t", "--test_function", type=str, help="Test function")
    parser.add_argument("-x", "--minx", nargs=2, type=float, help="x minimum values")
    parser.add_argument("-y", "--miny", nargs=2, type=float, help="y minimum values")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed for random")
    parser.add_argument("-l", "--leafs", type=int, default=4, help="Number of leafs")
    parser.add_argument("-n", "--nbexp", type=int, default=20, help="Range for number of experiments")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_whitebox_experiment_random_search(number_of_leafs=args.leafs,
                                          wallclock_limit=args.wallclock,
                                          seed=args.seed,
                                          test_function=args.test_function,
                                          nb_experiments=args.nbexp,
                                          min_x=args.minx,
                                          min_y=args.miny,
                                          output_dir=args.output_dir)