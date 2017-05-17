
# This file is heavily based on the pc_smbo file of SMAC which can be found here:
#   https://github.com/automl/SMAC3

import itertools
import math
import numpy as np
import logging
import typing
import time
import random

from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.base_solver import BaseSolver
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.local_search import LocalSearch
from smac.intensification.intensification import Intensifier
from smac.optimizer import pSMAC
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.stats.stats import Stats
from smac.initial_design.initial_design import InitialDesign
from smac.tae.execute_ta_run import FirstRunCrashedException
from smac.optimizer.select_configurations import SelectConfigurations


class PCSMBO(BaseSolver):

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 initial_design: InitialDesign,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 num_run: int,
                 model: RandomForestWithInstances,
                 acq_optimizer: LocalSearch,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 select_configuration: SelectConfigurations,
                 double_intensification: bool):
        '''
        Interface that contains the main Bayesian optimization loop

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: Stats
            statistics object with configuration budgets
        initial_design: InitialDesign
            initial sampling design
        runhistory: RunHistory
            runhistory with all runs so far
        runhistory2epm : AbstractRunHistory2EPM
            Object that implements the AbstractRunHistory2EPM to convert runhistory data into EPM data
        intensifier: Intensifier
            intensification of new challengers against incumbent configuration (probably with some kind of racing on the instances)
        aggregate_func: callable
            how to aggregate the runs in the runhistory to get the performance of a configuration
        num_run: int
            id of this run (used for pSMAC)
        model: RandomForestWithInstances
            empirical performance model (right now, we support only RandomForestWithInstances)
        acq_optimizer: LocalSearch
            optimizer on acquisition function (right now, we support only a local search)
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction (i.e., infill criterion for acq_optimizer)
        rng: np.random.RandomState
            Random number generator
        '''
        self.logger = logging.getLogger("SMBO")
        self.incumbent = None

        self.scenario = scenario
        self.config_space = scenario.cs
        self.stats = stats
        self.initial_design = initial_design
        self.runhistory = runhistory
        self.rh2EPM = runhistory2epm
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.num_run = num_run
        self.model = model
        self.acq_optimizer = acq_optimizer
        self.acquisition_func = acquisition_func
        self.rng = rng

        self.select_configuration = select_configuration
        self.double_intensification = double_intensification
        print("Initialize PCSMBO: double intensification: {}".format(self.double_intensification))

    def run(self):
        '''
        Runs the Bayesian optimization loop

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        '''
        self.stats.start_timing()
        try:
            self.incumbent = self.initial_design.run()
        except FirstRunCrashedException as err:
            if self.scenario.abort_on_first_run_crash:
                raise

        # Main BO loop
        iteration = 1
        while True:
            if self.scenario.shared_model:
                pSMAC.read(run_history=self.runhistory,
                           output_directory=self.scenario.output_dir,
                           configuration_space=self.config_space,
                           logger=self.logger)

            start_time = time.time()

            X, Y = self.rh2EPM.transform(self.runhistory)
            print("Shapes: {}, {}".format(X.shape, Y.shape))

            self.logger.debug("Search for next configuration")
            if self.double_intensification:
                # get all found configurations sorted according to acq
                challengers_smac, challengers_random = \
                    self.select_configuration.run(X, Y,
                                                  incumbent=self.incumbent,
                                                  num_configurations_by_random_search_sorted=100,
                                                  num_configurations_by_local_search=10,
                                                  double_intensification=self.double_intensification)

                time_spend = time.time() - start_time
                logging.debug(
                    "Time spend to choose next configurations: %.2f sec" % (time_spend))

                self.logger.debug("Intensify")

                start_time_random = time.time()
                self.incumbent, inc_perf = self.intensifier.intensify(
                    challengers=challengers_random,
                    incumbent=self.incumbent,
                    run_history=self.runhistory,
                    aggregate_func=self.aggregate_func,
                    time_bound=max(0.01, time_spend / 2.),
                    min_number_of_runs=1)
                time_spend_random = time.time() - start_time_random

                print("IN BETWEEN INTENSIFICATIONS")

                self.incumbent, inc_perf = self.intensifier.intensify(
                    challengers=challengers_smac,
                    incumbent=self.incumbent,
                    run_history=self.runhistory,
                    aggregate_func=self.aggregate_func,
                    time_bound=max(0.01, time_spend_random),
                    min_number_of_runs=1)
            else:
                # get all found configurations sorted according to acq
                challengers = \
                    self.select_configuration.run(X, Y,
                                                  incumbent=self.incumbent,
                                                  num_configurations_by_random_search_sorted=100,
                                                  num_configurations_by_local_search=10,
                                                  double_intensification=self.double_intensification)

                print([type(challenger) for challenger in challengers])
                print("double intensification: {}".format(self.double_intensification))
                #print([type(challenger) for challenger in challengers if str(type(challenger)) != "<class 'ConfigSpace.configuration_space.Configuration'>"])

                time_spend = time.time() - start_time
                logging.debug(
                    "Time spend to choose next configurations: %.2f sec" % (time_spend))

                self.logger.debug("Intensify")

                self.incumbent, inc_perf = self.intensifier.intensify(
                    challengers=challengers,
                    incumbent=self.incumbent,
                    run_history=self.runhistory,
                    aggregate_func=self.aggregate_func,
                    time_bound=max(0.01, time_spend),
                    min_number_of_runs=2)

            print("Incumbent: {}, Performance: {}".format(self.incumbent, inc_perf))

            if self.scenario.shared_model:
                pSMAC.write(run_history=self.runhistory,
                            output_directory=self.scenario.output_dir,
                            num_run=self.num_run)

            iteration += 1

            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent