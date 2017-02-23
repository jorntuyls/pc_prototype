
import typing
import time

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.utils.constants import MAXINT, MAX_CUTOFF
from smac.intensification.intensification import Intensifier

class PCIntensifier(Intensifier):

    def __init__(self, tae_runner, stats, traj_logger, rng, instances,
                 instance_specifics=None, cutoff=MAX_CUTOFF, deterministic=False,
                 run_obj_time=True, run_limit=MAXINT, minR=1, maxR=2000):
        super(PCIntensifier,self).__init__(tae_runner=tae_runner,
                                           stats=stats,
                                           traj_logger=traj_logger,
                                           rng=rng,
                                           instances=instances,
                                           instance_specifics=instance_specifics,
                                           cutoff=cutoff,
                                           deterministic=deterministic,
                                           run_obj_time=run_obj_time,
                                           run_limit=run_limit,
                                           minR=minR,
                                           maxR=maxR)

    def intensify(self, challengers: typing.List[Configuration],
                  incumbent: Configuration,
                  run_history: RunHistory,
                  aggregate_func: typing.Callable,
                  time_bound: int = MAXINT):
        '''
            running intensification to determine the incumbent configuration
            Side effect: adds runs to run_history

            Implementation of Procedure 2 in Hutter et al. (2011).

            Parameters
            ----------

            challengers : typing.List[Configuration]
                promising configurations
            incumbent : Configuration
                best configuration so far
            run_history : RunHistory
                stores all runs we ran so far
            aggregate_func: typing.Callable
                aggregate performance across instances
            time_bound : int, optional (default=2 ** 31 - 1)
                time in [sec] available to perform intensify

            Returns
            -------
            incumbent: Configuration()
                current (maybe new) incumbent configuration
            inc_perf: float
                empirical performance of incumbent configuration
        '''

        self.start_time = time.time()

        if time_bound < 0.01:
            raise ValueError("time_bound must be >= 0.01")

        self._num_run = 0
        self._chall_indx = 0

        # Line 1 + 2
        for challenger in challengers:
            if challenger == incumbent:
                self.logger.warning(
                    "Challenger was the same as the current incumbent; Skipping challenger")
                continue

            self.logger.debug("Intensify on %s", challenger)
            if hasattr(challenger, 'origin'):
                self.logger.debug(
                    "Configuration origin: %s", challenger.origin)

            # Lines 3-7
            self._add_inc_run(incumbent=incumbent, run_history=run_history)

            # Lines 8-17
            incumbent = self._race_challenger(challenger=challenger,
                                              incumbent=incumbent,
                                              run_history=run_history,
                                              aggregate_func=aggregate_func)

            if self._chall_indx > 1 and self._num_run > self.run_limit:
                self.logger.debug(
                    "Maximum #runs for intensification reached")
                break
            elif self._chall_indx > 1 and time.time() - self.start_time - time_bound >= 0:
                self.logger.debug("Timelimit for intensification reached ("
                                  "used: %f sec, available: %f sec)" % (
                                      time.time() - self.start_time, time_bound))
                break

        # output estimated performance of incumbent
        inc_runs = run_history.get_runs_for_config(incumbent)
        inc_perf = aggregate_func(incumbent, run_history, inc_runs)
        self.logger.info("Updated estimated performance of incumbent on %d runs: %.4f" % (
            len(inc_runs), inc_perf))

        self.stats.update_average_configs_per_intensify(
            n_configs=self._chall_indx)

        return incumbent, inc_perf
