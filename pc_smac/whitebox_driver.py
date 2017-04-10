import os
import time

from pc_smac.pc_smac.whitebox.paraboloid import Paraboloid, CachedParaboloid, CachedParaboloid2Minima, Paraboloid2Minima
from pc_smac.pc_smac.whitebox.beale_function import Beale, CachedBeale
from pc_smac.pc_smac.pc_smbo.smbo_builder import SMBOBuilder
from pc_smac.pc_smac.pc_runhistory.pc_runhistory import PCRunHistory
from pc_smac.pc_smac.utils.statistics import Statistics, WhiteboxStatistics
from pc_smac.pc_smac.utils.statistics_whitebox import WhiteboxStats
from pc_smac.pc_smac.random_search.whitebox_random_search import WhiteboxRandomSearch, Whitebox2stepRandomSearch

from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.smbo.objective import average_cost

class WhiteBoxDriver:

    def __init__(self, output_dir=None):

        self.output_dir = output_dir if output_dir else os.path.dirname(os.path.abspath(__file__)) + "/output/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initialize(self, stamp, seed, acq_func, wallclock_limit, runcount_limit, test_function, random_leaf_size, min_x, min_y):
        # Check if caching is enabled
        caching = True if acq_func[:2] == "pc" else False

        # Build runhistory
        # TODO Does this work correctly for non-caching?
        runhistory = PCRunHistory(average_cost)

        # Setup statistics
        info = {
            'stamp': stamp,
            'caching': caching,
            'acquisition_function': acq_func,
            'wallclock_limit': wallclock_limit
        }

        self.statistics = WhiteboxStatistics(stamp,
                                     self.output_dir,
                                     information=info,
                                     total_runtime=wallclock_limit)

        # Set up tae runner
        if caching:
            if test_function == "beale":
                tae = CachedBeale(runhistory=runhistory,
                                  statistics=self.statistics,
                                  min_x=min_x,
                                  min_y=min_y)
            else:
                tae = CachedParaboloid2Minima(runhistory=runhistory,
                                              statistics=self.statistics,
                                              min_x=min_x,
                                              min_y=min_y)
        else:
            if test_function == "beale":
                tae = Beale(runhistory=runhistory,
                            statistics=self.statistics,
                            min_x=min_x,
                            min_y=min_y)
            else:
                tae = Paraboloid2Minima(runhistory=runhistory,
                                        statistics=self.statistics,
                                        min_x=min_x,
                                        min_y=min_y)

        # setup config space
        self.config_space = tae.get_config_space(seed=seed)

        # Build scenario
        args = {'cs': self.config_space,
                'run_obj': "quality",
                'wallclock_limit': wallclock_limit,
                'runcount_limit': runcount_limit,
                'deterministic': "true"
                }
        scenario = Scenario(args)

        # Build stats
        stats = WhiteboxStats(scenario,
                                output_dir=self.output_dir + "/smac/",
                                stamp=stamp)
        # Give the stats to the tae runner to simulate timing
        tae.set_smac_stats(stats)

        # Choose acquisition function
        if acq_func in ["eips", "m-eips", "pc-m-eips", "pceips", "pc-m-pceips"]:
            model_target_names = ['cost', 'time']
        elif acq_func in ["ei", "m-ei", "pc-m-ei"]:
            model_target_names = ['cost']
        else:
            # Not a valid acquisition function
            raise ValueError("The provided acquisition function is not valid")

        # Setup trajectory file
        trajectory_path = self.output_dir + "/logging/" + stamp + "/" # + self.data_path.split("/")[-1] + "/" + str(stamp)
        if not os.path.exists(trajectory_path):
            os.makedirs(trajectory_path)
        self.trajectory_path_json = trajectory_path + "/traj_aclib2.json"
        self.trajectory_path_csv = trajectory_path + "/traj_old.csv"

        # Build SMBO object
        smbo_builder = SMBOBuilder()
        self.smbo = smbo_builder.build_pc_smbo(
            tae_runner=tae,
            stats=stats,
            scenario=scenario,
            runhistory=runhistory,
            aggregate_func=average_cost,
            acq_func_name=acq_func,
            model_target_names=model_target_names,
            logging_directory=trajectory_path,
            random_leaf_size=random_leaf_size,
            constant_pipeline_steps=["preprocessor"],
            variable_pipeline_steps=["classifier"])


    def run(self,
            stamp=time.time(),
            seed=None,
            acq_func="ei",
            wallclock_limit=3600,
            runcount_limit=1000,
            test_function="paraboloid",
            random_leaf_size=4,
            min_x=0.75,
            min_y=0.5):

        # Initialize SMBO
        self.initialize(stamp=stamp,
                        seed=seed,
                        acq_func=acq_func,
                        wallclock_limit=wallclock_limit,
                        runcount_limit=runcount_limit,
                        test_function=test_function,
                        random_leaf_size=random_leaf_size,
                        min_x=min_x,
                        min_y=min_y)

        # clean trajectory files
        self._clean_trajectory_files()

        # Start timer and clean statistics files
        self.statistics.start_timer()
        self.statistics.clean_files()

        # Run SMBO
        incumbent = self.smbo.run()

        # Read trajectory files with incumbents and retrieve test performances
        trajectory = TrajLogger.read_traj_aclib_format(self.trajectory_path_json, self.config_space)
        # First transform the configuration to a dictionary
        for traj in trajectory:
            traj['incumbent'] = traj['incumbent'].get_dictionary()
        self.statistics.add_incumbents_trajectory(trajectory)
        return incumbent

    #### INTERNAL METHODS ####

    def _clean_trajectory_files(self):
        open(self.trajectory_path_json, 'w')
        open(self.trajectory_path_csv, 'w')



class WhiteboxDriverRandomSearch(object):

    def __init__(self, output_dir=None):
        self.output_dir = output_dir if output_dir else os.path.dirname(os.path.abspath(__file__)) + "/output/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run(self,
          stamp,
          version,
          number_of_leafs,
          wallclock_limit,
          seed=None,
          test_function="paraboloid",
          min_x=[0.90, 0.90],
          min_y=[0.1, 0.9]):

        # Build statistics
        info = {
            'version': version,
            'wallclock_limit': wallclock_limit,
            'seed': seed,
        }
        self.statistics = WhiteboxStatistics(stamp,
                                        self.output_dir,
                                        information=info,
                                        total_runtime=wallclock_limit)

        self.statistics.clean_files()

        if version == '2step':
            if test_function == "beale":
                tae = CachedBeale(runhistory=None,
                                  statistics=self.statistics,
                                  min_x=min_x,
                                  min_y=min_y)
            else:
                tae = CachedParaboloid2Minima(runhistory=None,
                                              statistics=self.statistics,
                                              min_x=min_x,
                                              min_y=min_y)
            # Build configuration space
            config_space = tae.get_config_space(seed=seed)
            random_search = Whitebox2stepRandomSearch(config_space=config_space,
                                                      pipeline_runner=tae,
                                                      wallclock_limit=wallclock_limit,
                                                      statistics=self.statistics,
                                                      constant_pipeline_steps=["preprocessor"],
                                                      variable_pipeline_steps=["classifier"],
                                                      number_leafs_split=number_of_leafs)
        else:
            if test_function == "beale":
                tae = Beale(runhistory=None,
                            statistics=self.statistics,
                            min_x=min_x,
                            min_y=min_y)
            else:
                tae = Paraboloid2Minima(runhistory=None,
                                        statistics=self.statistics,
                                        min_x=min_x,
                                        min_y=min_y)
            # Build configuration space
            config_space = tae.get_config_space(seed=seed)
            random_search = WhiteboxRandomSearch(config_space=config_space,
                                                 pipeline_runner=tae,
                                                 wallclock_limit=wallclock_limit,
                                                 statistics=self.statistics)

        # Run random search
        print("start random search")
        incumbent = random_search.run()
        print("... end random search")

        trajectory = self.statistics.get_incumbent_trajectory_dicts()
        self.statistics.add_incumbents_trajectory(trajectory=trajectory)

        return incumbent
