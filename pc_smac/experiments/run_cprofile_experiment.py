
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
from smac.utils.util_funcs import get_types
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.optimizer.objective import average_cost

from pc_smac.pc_smac.config_space.config_space_builder import ConfigSpaceBuilder
from pc_smac.pc_smac.pipeline_space.pipeline_step import OneHotEncodingStep, ImputationStep, RescalingStep, \
    BalancingStep, PreprocessingStep, ClassificationStep
from pc_smac.pc_smac.pipeline_space.pipeline_space import PipelineSpace
from pc_smac.pc_smac.pc_runhistory.pc_runhistory import PCRunHistory



def run_experiment():
    pipeline_space = PipelineSpace()
    o_s = OneHotEncodingStep()
    i_s = ImputationStep()
    r_s = RescalingStep()
    b_s = BalancingStep()
    p_s = PreprocessingStep()
    c_s = ClassificationStep()
    pipeline_space.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s])

    runhistory = PCRunHistory(average_cost)

    cs_builder = ConfigSpaceBuilder(pipeline_space)
    config_space = cs_builder.build_config_space()

    args = {'cs': config_space,
            'run_obj': "quality",
            'runcount_limit': 100,
            'wallclock_limit': 100,
            'memory_limit': 100,
            'cutoff_time': 100,
            'deterministic': "true"
            }
    scenario = Scenario(args)

    # Build stats
    stats = Stats(scenario,
                  output_dir=None,
                  stamp="")

    types, bounds = get_types(scenario.cs, scenario.feature_array)

    model = RandomForestWithInstances(types=types, bounds=bounds)

    constant_pipeline_steps = ["one_hot_encoder", "imputation", "rescaling",
                               "balancing", "feature_preprocessor"]
    variable_pipeline_steps = ["classifier"]
    rng = np.random.RandomState()
    num_params = len(scenario.cs.get_hyperparameters())

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
    select_configuration = SelectConfigurationsWithMarginalization(scenario=scenario,
                                                                   stats=stats,
                                                                   runhistory=runhistory,
                                                                   model=model,
                                                                   acq_optimizer=local_search,
                                                                   acquisition_func=acq_func_wrapper,
                                                                   rng=rng,
                                                                   constant_pipeline_steps=constant_pipeline_steps,
                                                                   variable_pipeline_steps=variable_pipeline_steps,
                                                                   num_marginalized_configurations_by_random_search=40,
                                                                   num_configs_for_marginalization=200)

    # sample configurations to fill runhistory
    sample_configs = config_space.sample_configuration(size=10)
    for config in sample_configs:
        runhistory.add(config, 1, 1, StatusType.SUCCESS)

    # test select_configurations procedure
    X, Y = runhistory2epm.transform(runhistory)
    challengers = select_configuration.run(X, Y,
                                           sample_configs[0],
                                           num_configurations_by_random_search_sorted=100,
                                           num_configurations_by_local_search=10,
                                           random_leaf_size=1)

    print(challengers[0])


if __name__ == "__main__":
    run_experiment()