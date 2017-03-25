
import argparse
from pc_smac.pc_smac.whitebox_driver import WhiteBoxDriver

def run_whitebox_experiment_pceips(wallclock_limit, seed=None, nb_experiments=20, test_function = "paraboloid", min_x = [0.90, 0.90], min_y = [0.1, 0.9]):
    # Setup preferences
    acq_funcs = ['pceips']
    random_leaf_sizes = [1] #,2,3,4,5,6,7,8,9,10,20,50]
    for acq_func in acq_funcs:
        # Initialize whitebox driver
        for rls in random_leaf_sizes:
            output_dir = '/Users/jorntuyls/Documents/workspaces/thesis/pc_smac/output/{}'.format(acq_func + "_" + str(rls))
            for i in range(0, nb_experiments):
                wd = WhiteBoxDriver(output_dir=output_dir)
                incumbent = wd.run(stamp=acq_func + "_" + str(rls) + "_run_" + str(i),
                                   seed=seed,
                                   acq_func=acq_func,
                                   wallclock_limit=wallclock_limit,
                                   runcount_limit=1,
                                   test_function=test_function,
                                   random_leaf_size=rls,
                                   min_x=min_x,
                                   min_y=min_y)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=str, help="Random search version")
    parser.add_argument("-w", "--wallclock", type=int, help="Wallclock limit")
    parser.add_argument("-t", "--test_function", type=str, help="Test function")
    parser.add_argument("-x", "--minx", nargs=2, type=float, help="x minimum values")
    parser.add_argument("-y", "--miny", nargs=2, type=float, help="y minimum values")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed for random")
    parser.add_argument("-n", "--nbexp", type=int, default=20, help="Range for number of experiments")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_whitebox_experiment_pceips(wallclock_limit=args.wallclock,
                                   seed=args.seed,
                                   test_function=args.test_function,
                                   nb_experiments=args.nbexp,
                                   min_x=args.minx,
                                   min_y=args.miny)