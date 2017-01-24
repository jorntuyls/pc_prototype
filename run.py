
import argparse

from pc_smac.pc_smac.pc_driver import Driver


def run_smac(acq_func, wallclock_limit, data_path, stamp, output_dir, cache_directory, downsampling):
    d = Driver(data_path=data_path, output_dir=output_dir)
    return d.run(acq_func=acq_func,
                 wallclock_limit=wallclock_limit,
                 stamp=stamp,
                 cache_directory=cache_directory,
                 downsampling=downsampling)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--acquisition", type=str, help="Acquisition value, in ['ei', 'eips', 'pceips']")
    parser.add_argument("-w", "--wallclock", type=int, help="Wallclock limit")
    parser.add_argument("-l", "--location", type=str, help="Data location")
    parser.add_argument("-s", "--stamp", type=str, default="stamp", help="Stamp for output files")
    parser.add_argument("-o", "--outputdir", type=str, default=None, help="Output directory")
    parser.add_argument("-cd", "--cachedir", type=str, default=None, help="Cache directory")
    parser.add_argument("-ds", "--downsampling", type=int, default=None, help="Downsampling of data")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_smac(args.acquisition, args.wallclock, args.location, args.stamp, args.outputdir, args.cachedir, args.downsampling)

