
import argparse

from pc_driver import Driver

#from pc_prototype.pc_smac.utils.io_utils import save_info_file
from pc_smac.pc_smac.data_paths import data_path, cache_directory


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
                   time_precision=plot_time)
        incs.append(inc)
    info = "Name: {}\n caching: {)\n cache_directory: {}\n wallclock_limit: {}\n downsampling: {}\n " \
           "totalruns: {}\n plot_time: {}".format(name, caching, cache_directory, wallclock_limit, downsampling,
                                                 total_runs, plot_time)

    #save_info_file(info, filename=name)
elif experiment_number == 2:
    name = "Experiment_2"
    d = Driver(data_path)
    incs = []
    #
    total_runs = 1
    for i in range(0, total_runs):
        inc= d.run(
                   stamp="experiment2",
                   caching=False,
                   cache_directory=cache_directory,
                   wallclock_limit=10,
                   downsampling=2000,
                   run_counter=i,
                   time_precision=10)
        incs.append(inc)

