
import numpy as np

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory2epm import RunHistory2EPM4EIPS, RunHistory2EPM4Cost
from smac.smbo.acquisition import EI, EIPS
from smac.smbo.local_search import LocalSearch
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.uncorrelated_mo_rf_with_instances import UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.intensification.intensification import Intensifier
from smac.initial_design.random_configuration_design import RandomConfiguration
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.util_funcs import get_types
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict

from pc_smac.pc_smac.pc_smbo.pc_smbo import PCSMBO
from pc_smac.pc_smac.pc_smbo.pc_acquisition import PCEIPS
from pc_smac.pc_smac.pc_smbo.pc_local_search import PCLocalSearch
from pc_smac.pc_smac.pc_smbo.select_configuration import CachedSelectConfiguration, SelectConfiguration



class SMBOBuilder:

    def __init__(self):
        pass

    def build_pc_smbo(self, execute_func, scenario, runhistory, aggregate_func, acq_func_name, model_target_names,
                        logging_directory):

        # Build stats
        stats = Stats(scenario)

        # Build tae runner
        tae_runner = ExecuteTAFuncDict(ta=execute_func,
                                       stats=stats,
                                       runhistory=runhistory,
                                       run_obj=scenario.run_obj,
                                       memory_limit=scenario.memory_limit)


        # Build intensifier
        rng = np.random.RandomState()
        traj_logger = TrajLogger(logging_directory, stats)
        intensifier = Intensifier(tae_runner=tae_runner,
                                  stats=stats,
                                  traj_logger=traj_logger,
                                  rng=rng,
                                  cutoff= scenario.cutoff,
                                  deterministic=scenario.deterministic,
                                  run_obj_time=scenario.run_obj == "runtime",
                                  run_limit=scenario.ta_run_limit,
                                  instances=[1],
                                  maxR=1)

        # Build model
        if len(model_target_names) > 1:
            # model_target_names = ['cost','time']
            model = UncorrelatedMultiObjectiveRandomForestWithInstances(target_names=model_target_names,
                                                                        types=get_types(scenario.cs))
        else:
            model = RandomForestWithInstances(get_types(scenario.cs))

        # Build acquisition function, runhistory2epm and local search
        num_params = len(scenario.cs.get_hyperparameters())
        if acq_func_name == 'eips':
            acquisition_func = EIPS(model)
            runhistory2epm = RunHistory2EPM4EIPS(scenario,
                                                 num_params,
                                                 success_states=[StatusType.SUCCESS])
            local_search = LocalSearch(acquisition_function=acquisition_func,
                                       config_space=scenario.cs)
            select_configuration = SelectConfiguration(scenario=scenario,
                                                       stats=stats,
                                                       runhistory=runhistory,
                                                       model=model,
                                                       acq_optimizer=local_search,
                                                       acquisition_func=acquisition_func,
                                                       rng=rng)

        elif acq_func_name == 'pceips':
            acquisition_func = PCEIPS(model)
            runhistory2epm = RunHistory2EPM4EIPS(scenario, num_params, success_states=[StatusType.SUCCESS])
            local_search = PCLocalSearch(acquisition_function=acquisition_func,
                                         config_space=scenario.cs)
            select_configuration = CachedSelectConfiguration(scenario=scenario,
                                                       stats=stats,
                                                       runhistory=runhistory,
                                                       model=model,
                                                       acq_optimizer=local_search,
                                                       acquisition_func=acquisition_func,
                                                       rng=rng)
        else:
            acquisition_func = EI(model)
            runhistory2epm = RunHistory2EPM4Cost(scenario, num_params,
                                                 success_states=[StatusType.SUCCESS])
            local_search = LocalSearch(acquisition_function=acquisition_func,
                                       config_space=scenario.cs)
            select_configuration = SelectConfiguration(scenario=scenario,
                                                       stats=stats,
                                                       runhistory=runhistory,
                                                       model=model,
                                                       acq_optimizer=local_search,
                                                       acquisition_func=acquisition_func,
                                                       rng=rng)


        # Build initial design
        initial_design = RandomConfiguration(tae_runner=tae_runner,
                                             scenario=scenario,
                                             stats=stats,
                                             traj_logger=traj_logger,
                                             rng=rng)

        # run id
        num_run = rng.randint(1234567980)

        # Build pc_smbo
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
                      rng=rng,
                      select_configuration=select_configuration)

        return smbo
