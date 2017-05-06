
import numpy as np

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory2epm import RunHistory2EPM4EIPS, RunHistory2EPM4Cost
from smac.optimizer.acquisition import EI, EIPS, PCEIPS
from smac.optimizer.local_search import LocalSearch
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.uncorrelated_mo_rf_with_instances import UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.intensification.intensification import Intensifier
from smac.optimizer.select_configurations import SelectConfigurations, SelectConfigurationsWithMarginalization
from smac.optimizer.acquisition_func_wrapper import PCAquisitionFunctionWrapper, PCAquisitionFunctionWrapperWithCachingReduction
from smac.initial_design.random_configuration_design import RandomConfiguration
from smac.initial_design.multi_config_initial_design import MultiConfigInitialDesign
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.util_funcs import get_types

from pc_smac.pc_smac.pc_smbo.pc_smbo import PCSMBO



class SMBOBuilder:

    def __init__(self):
        pass

    def build_pc_smbo(self, tae_runner, stats, scenario, runhistory, aggregate_func, acq_func_name, model_target_names,
                        logging_directory, mrs=False, constant_pipeline_steps=None, variable_pipeline_steps=None,
                      cached_pipeline_steps=None,
                      intensification_instances=None, num_marginalized_configurations_by_random_search=20, num_configs_for_marginalization=40):

        # Build intensifier
        rng = np.random.RandomState()
        traj_logger = TrajLogger(logging_directory, stats)
        intensifier = Intensifier(tae_runner=tae_runner,
                                  stats=stats,
                                  traj_logger=traj_logger,
                                  rng=rng,
                                  cutoff=scenario.cutoff,
                                  deterministic=scenario.deterministic,
                                  run_obj_time=scenario.run_obj == "runtime",
                                  run_limit=scenario.ta_run_limit,
                                  instances=intensification_instances,
                                  maxR=len(intensification_instances))

        # Build model
        types, bounds = get_types(scenario.cs, scenario.feature_array)
        #types = get_types(scenario.cs)
        if len(model_target_names) > 1:
            # model_target_names = ['cost','time']
            model = UncorrelatedMultiObjectiveRandomForestWithInstances(target_names=model_target_names,
                                                                       bounds=bounds,
                                                                       types=types)
            # UncorrelatedMultiObjectiveRandomForestWithInstances(target_names=model_target_names,
            #                                                    types=types)
        else:
            model = RandomForestWithInstances(types=types, bounds=bounds)
            # model = RandomForestWithInstances(types=types)

        # Build acquisition function, runhistory2epm and local search
        num_params = len(scenario.cs.get_hyperparameters())
        if acq_func_name == "ei":
            acquisition_func = EI(model)
            acq_func_wrapper = PCAquisitionFunctionWrapper(acquisition_func=acquisition_func,
                                                           config_space=scenario.cs,
                                                           runhistory=runhistory,
                                                           constant_pipeline_steps=constant_pipeline_steps,
                                                           variable_pipeline_steps=variable_pipeline_steps)
            runhistory2epm = RunHistory2EPM4Cost(scenario, num_params,
                                                 success_states=[StatusType.SUCCESS])
            local_search = LocalSearch(acquisition_function=acq_func_wrapper,
                                       config_space=scenario.cs)
            select_configuration = SelectConfigurations(scenario=scenario,
                                                        stats=stats,
                                                        runhistory=runhistory,
                                                        model=model,
                                                        acq_optimizer=local_search,
                                                        acquisition_func=acq_func_wrapper,
                                                        rng=rng,
                                                        constant_pipeline_steps=constant_pipeline_steps,
                                                        variable_pipeline_steps=variable_pipeline_steps)
        elif acq_func_name in ["m-ei", "pc-m-ei"]:
            #acquisition_func = MEI(model)
            acquisition_func = EI(model)
            acq_func_wrapper = PCAquisitionFunctionWrapper(acquisition_func=acquisition_func,
                                                    config_space=scenario.cs,
                                                    runhistory=runhistory,
                                                    constant_pipeline_steps=constant_pipeline_steps,
                                                    variable_pipeline_steps=variable_pipeline_steps)
            runhistory2epm = RunHistory2EPM4Cost(scenario, num_params,
                                                 success_states=[StatusType.SUCCESS])
            local_search = LocalSearch(acquisition_function=acq_func_wrapper,
                                         config_space=scenario.cs)
            # TODO: num_configs_for_marginalization
            select_configuration = SelectConfigurationsWithMarginalization(scenario=scenario,
                                                                          stats=stats,
                                                                          runhistory=runhistory,
                                                                          model=model,
                                                                          acq_optimizer=local_search,
                                                                          acquisition_func=acq_func_wrapper,
                                                                          rng=rng,
                                                                          constant_pipeline_steps=constant_pipeline_steps,
                                                                          variable_pipeline_steps=variable_pipeline_steps,
                                                                          num_marginalized_configurations_by_random_search=num_marginalized_configurations_by_random_search,
                                                                          num_configs_for_marginalization=num_configs_for_marginalization)
        elif acq_func_name == 'eips':
            acquisition_func = EIPS(model)
            acq_func_wrapper = PCAquisitionFunctionWrapper(acquisition_func=acquisition_func,
                                                    config_space=scenario.cs,
                                                    runhistory=runhistory,
                                                    constant_pipeline_steps=constant_pipeline_steps,
                                                    variable_pipeline_steps=variable_pipeline_steps)
            runhistory2epm = RunHistory2EPM4EIPS(scenario,
                                                 num_params,
                                                 success_states=[StatusType.SUCCESS])
            local_search = LocalSearch(acquisition_function=acq_func_wrapper,
                                       config_space=scenario.cs)
            select_configuration = SelectConfigurations(scenario=scenario,
                                                       stats=stats,
                                                       runhistory=runhistory,
                                                       model=model,
                                                       acq_optimizer=local_search,
                                                       acquisition_func=acq_func_wrapper,
                                                       rng=rng,
                                                       constant_pipeline_steps=constant_pipeline_steps,
                                                       variable_pipeline_steps=variable_pipeline_steps)
        elif acq_func_name in ["m-eips", "pc-m-eips"]:
            acquisition_func = EIPS(model)
            acq_func_wrapper = PCAquisitionFunctionWrapper(acquisition_func=acquisition_func,
                                                    config_space=scenario.cs,
                                                    runhistory=runhistory,
                                                    constant_pipeline_steps=constant_pipeline_steps,
                                                    variable_pipeline_steps=variable_pipeline_steps)
            runhistory2epm = RunHistory2EPM4EIPS(scenario,
                                                 num_params,
                                                 success_states=[StatusType.SUCCESS])
            local_search = LocalSearch(acquisition_function=acq_func_wrapper,
                                       config_space=scenario.cs)
            # TODO: num_configs_for_marginalization
            select_configuration = SelectConfigurationsWithMarginalization(scenario=scenario,
                                                                          stats=stats,
                                                                          runhistory=runhistory,
                                                                          model=model,
                                                                          acq_optimizer=local_search,
                                                                          acquisition_func=acq_func_wrapper,
                                                                          rng=rng,
                                                                          constant_pipeline_steps=constant_pipeline_steps,
                                                                          variable_pipeline_steps=variable_pipeline_steps,
                                                                          num_marginalized_configurations_by_random_search=num_marginalized_configurations_by_random_search,
                                                                          num_configs_for_marginalization=num_configs_for_marginalization)
        elif acq_func_name == 'pceips':
            acquisition_func = PCEIPS(model)
            acq_func_wrapper = PCAquisitionFunctionWrapperWithCachingReduction(acquisition_func=acquisition_func,
                                                                        config_space=scenario.cs,
                                                                        runhistory=runhistory,
                                                                        constant_pipeline_steps=constant_pipeline_steps,
                                                                        variable_pipeline_steps=variable_pipeline_steps,
                                                                               cached_pipeline_steps=cached_pipeline_steps)
            runhistory2epm = RunHistory2EPM4EIPS(scenario, num_params, success_states=[StatusType.SUCCESS])
            local_search = LocalSearch(acquisition_function=acq_func_wrapper,
                                         config_space=scenario.cs)
            if constant_pipeline_steps == None or variable_pipeline_steps == None or cached_pipeline_steps == None:
                raise ValueError("Constant_pipeline_steps and variable pipeline steps should not be none\
                                    when using PCEIPS")
            select_configuration = SelectConfigurations(scenario=scenario,
                                                       stats=stats,
                                                       runhistory=runhistory,
                                                       model=model,
                                                       acq_optimizer=local_search,
                                                       acquisition_func=acq_func_wrapper,
                                                       rng=rng,
                                                       constant_pipeline_steps=constant_pipeline_steps,
                                                       variable_pipeline_steps=variable_pipeline_steps)
        elif acq_func_name == 'pc-m-pceips':
            acquisition_func = PCEIPS(model)
            acq_func_wrapper = PCAquisitionFunctionWrapperWithCachingReduction(acquisition_func=acquisition_func,
                                                                        config_space=scenario.cs,
                                                                        runhistory=runhistory,
                                                                        constant_pipeline_steps=constant_pipeline_steps,
                                                                        variable_pipeline_steps=variable_pipeline_steps,
                                                                               cached_pipeline_steps=cached_pipeline_steps)
            runhistory2epm = RunHistory2EPM4EIPS(scenario, num_params, success_states=[StatusType.SUCCESS])
            local_search = LocalSearch(acquisition_function=acq_func_wrapper,
                                         config_space=scenario.cs)
            if constant_pipeline_steps == None or variable_pipeline_steps == None or cached_pipeline_steps == None:
                raise ValueError("Constant_pipeline_steps and variable pipeline steps should not be none\
                                    when using PCEIPS")
            select_configuration = SelectConfigurationsWithMarginalization(scenario=scenario,
                                                                          stats=stats,
                                                                          runhistory=runhistory,
                                                                          model=model,
                                                                          acq_optimizer=local_search,
                                                                          acquisition_func=acq_func_wrapper,
                                                                          rng=rng,
                                                                          constant_pipeline_steps=constant_pipeline_steps,
                                                                          variable_pipeline_steps=variable_pipeline_steps,
                                                                          num_marginalized_configurations_by_random_search=num_marginalized_configurations_by_random_search,
                                                                          num_configs_for_marginalization=num_configs_for_marginalization)
        else:
            # Not a valid acquisition function
            raise ValueError("The provided acquisition function is not valid")



        # Build initial design
        # initial_design = RandomConfiguration(tae_runner=tae_runner,
        #                                      scenario=scenario,
        #                                      stats=stats,
        #                                      traj_logger=traj_logger,
        #                                      rng=rng)
        initial_configs = scenario.cs.sample_configuration(size=2)
        for config in initial_configs:
            config._populate_values()
        initial_design = MultiConfigInitialDesign(tae_runner=tae_runner,
                                                  scenario=scenario,
                                                  stats=stats,
                                                  traj_logger=traj_logger,
                                                  runhistory=runhistory,
                                                  rng=rng,
                                                  configs=initial_configs,
                                                  intensifier=intensifier,
                                                  aggregate_func=aggregate_func)

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
                      select_configuration=select_configuration,
                      mrs=mrs)

        return smbo
