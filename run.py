
import argparse

from pc_smac.pc_smac.pc_driver import Driver


def run_smac(acq_func, mrs, wallclock_limit, runcount_limit, memory_limit, cutoff, data_path, stamp, output_dir, cache_directory,
             downsampling, intensification_fold_size, pipeline_space_string):
    d = Driver(data_path=data_path, output_dir=output_dir, pipeline_space_string=pipeline_space_string)
    return d.run(stamp=stamp,
                 acq_func=acq_func,
                 mrs=mrs,
                 wallclock_limit=wallclock_limit,
                 runcount_limit=runcount_limit,
                 memory_limit=memory_limit,
                 cutoff=cutoff,
                 cache_directory=cache_directory,
                 downsampling=downsampling,
                 intensification_fold_size=intensification_fold_size)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--acquisition", type=str, help="Acquisition value, in ['ei', 'eips', 'pceips']")
    parser.add_argument("-mrs", "--multi_step_random_search", type=bool, default=False, help="Is multi-step random search enabled")
    parser.add_argument("-w", "--wallclock", type=int, help="Wallclock limit")
    parser.add_argument("-r", "--runlimit", type=int, default=10000, help="Limitation of the number of runs")
    parser.add_argument("-m", "--memory", type=int, default=6000, help="Memory limit")
    parser.add_argument("-c", "--cutoff", type=int, default=3600, help="Cutoff time for one run")
    parser.add_argument("-l", "--location", type=str, help="Data location")
    parser.add_argument("-s", "--stamp", type=str, default="stamp", help="Stamp for output files")
    parser.add_argument("-o", "--outputdir", type=str, default=None, help="Output directory")
    parser.add_argument("-cd", "--cachedir", type=str, default=None, help="Cache directory")
    parser.add_argument("-ds", "--downsampling", type=int, default=None, help="Downsampling of data")
    parser.add_argument("-ifs", "--intensification_fold_size", type=int, default=None, help="Intensification fold size")
    parser.add_argument("-ps", "--pipeline_space", type=str, default=None, help="Scenario to execute")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_smac(args.acquisition,
             args.multi_step_random_search,
             args.wallclock,
             args.runlimit,
             args.memory,
             args.cutoff,
             args.location,
             args.stamp,
             args.outputdir,
             args.cachedir,
             args.downsampling,
             args.intensification_fold_size,
             args.pipeline_space)


