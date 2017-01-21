
import argparse

from pc_driver import Driver

from utils.io_utils import save_info_file
from data_paths import data_path, cache_directory


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=int, help="Experiment number")
args = parser.parse_args()

experiment_number = args.exp
'''
Experiment number 1
-----------------------
Run driver multiple time with different run counter
'''
if experiment_number == 1:
    name = "Experiment_1"
    d = Driver(data_path)
    incs = []
    #
    caching = True
    wallclock_limit = 3600
    downsampling = 5000
    plot_time = 100
    #
    total_runs = 5
    for i in range(1, total_runs):
        inc= d.run(caching=caching,
                   cache_directory=cache_directory,
                   wallclock_limit=wallclock_limit,
                   downsampling=downsampling,
                   run_counter=i,
                   plot_time=plot_time)
        incs.append(inc)
    info = "Name: {}\n caching: {)\n cache_directory: {}\n wallclock_limit: {}\n downsampling: {}\n " \
           "totalruns: {}\n plot_time: {}".format(name, caching, cache_directory, wallclock_limit, downsampling,
                                                 total_runs, plot_time)

    save_info_file(info, filename=name)
elif experiment_number == 2:
    name = "Experiment_2"
    d = Driver(data_path)
    incs = []
    #
    caching = False
    wallclock_limit = 1080
    downsampling = 2000
    plot_time = 100
    #
    total_runs = 1
    for i in range(0, total_runs):
        inc= d.run(caching=caching,
                   cache_directory=cache_directory,
                   wallclock_limit=wallclock_limit,
                   downsampling=downsampling,
                   run_counter=i,
                   plot_time=plot_time)
        incs.append(inc)
    info = "Name: {}\n caching: {}\n cache_directory: {}\n wallclock_limit: {}\n downsampling: {}\n " \
           "totalruns: {}\n plot_time: {}".format(name, caching, cache_directory, wallclock_limit, downsampling,
                                                 total_runs, plot_time)

    save_info_file(info, filename=name)

