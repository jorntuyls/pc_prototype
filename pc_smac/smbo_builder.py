
import numpy as np

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.smbo.acquisition import EI
from smac.smbo.local_search import LocalSearch
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.scenario.scenario import Scenario
from smac.intensification.intensification import Intensifier
from smac.initial_design.random_configuration_design import RandomConfiguration
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.util_funcs import get_types
from smac.stats.stats import Stats

from pc_smbo import PCSMBO



class SMBOBuilder:

    def __init__(self):
        pass

    def build_pc_smbo(self, config_space, tae_runner, runhistory, aggregate_func,
                        wallclock_limit=60):
        # Build scenario
        args = {'cs': config_space, 'wallclock_limit': wallclock_limit}
        scenario = Scenario(args)

        # Build intensifier
        rng = np.random.RandomState()
        stats = Stats(scenario)
        traj_logger = TrajLogger("logging", stats)
        intensifier = Intensifier(tae_runner, stats, traj_logger, rng, maxR=1)

        # Build model
        model = RandomForestWithInstances(get_types(config_space))

        # Build acquisition function
        acquisition_func = EI(model)

        # Build runhistory2epm
        num_params = len(scenario.cs.get_hyperparameters())
        runhistory2epm = RunHistory2EPM4Cost(scenario, num_params, success_states=[StatusType.SUCCESS])

        # Build initial design
        initial_design = RandomConfiguration(tae_runner=tae_runner,
                         scenario=scenario,
                         stats=stats,
                         traj_logger=traj_logger,
                         runhistory=runhistory,
                         rng=rng)

        # run id
        num_run = rng.randint(1234567980)

        # Build local search
        local_search = LocalSearch(acquisition_function=acquisition_func,
                                    config_space=config_space)

        # Build smbo
        smbo = PCSMBO(scenario=scenario,
                        stats=stats,
                        initial_design=initial_design,
                        runhistory=runhistory,
                        runhistory2epm=runhistory2epm,
                        intensifier=intensifier,
                        aggregate_func=aggregate_func,
                        num_run=num_run,
                        model=model,
                        acq_optimizer=local_search,
                        acquisition_func=acquisition_func,
                        rng=rng)

        return smbo
