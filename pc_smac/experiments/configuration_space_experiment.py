
import traceback
import sys

import time
import numpy as np

from ConfigSpace.configuration_space import Configuration
from pc_smac.pc_smac.config_space.config_space_builder import ConfigSpaceBuilder
from pc_smac.pc_smac.pipeline_space.pipeline_step import OneHotEncodingStep, ImputationStep, RescalingStep, \
    BalancingStep, PreprocessingStep, ClassificationStep
from pc_smac.pc_smac.pipeline_space.pipeline_space import PipelineSpace


#### USE PCP18 ENVIRONMENT !!!

def run_experiment_vector():
    pipeline_space = PipelineSpace()
    o_s = OneHotEncodingStep()
    i_s = ImputationStep()
    r_s = RescalingStep()
    b_s = BalancingStep()
    p_s = PreprocessingStep()
    c_s = ClassificationStep()
    pipeline_space.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s])
    constant_pipeline_steps = ["one_hot_encoder", "imputation", "rescaling",
                               "balancing", "feature_preprocessor"]
    variable_pipeline_steps = ["classifier"]

    cs_builder = ConfigSpaceBuilder(pipeline_space)
    config_space = cs_builder.build_config_space()

    timing_v_1 = []
    timing_v_2 = []
    for i in range(0, 20):
        print("Run: {}".format(i))
        # sample 1 start config
        start_config = config_space.sample_configuration(size=1)
        sample_configs = config_space.sample_configuration(size=500)
        sample_configs_values = [get_values(evaluation_config.get_dictionary(), variable_pipeline_steps) \
                                 for evaluation_config in sample_configs]

        # version 1
        start_time = time.time()
        # new_configurations = combine_configurations_batch_version1(config_space=config_space,
        #                                                            start_config=start_config,
        #                                                            complemented_configs_values=sample_configs_values,
        #                                                            constant_pipeline_steps=constant_pipeline_steps)
        for config in sample_configs:
            config.is_valid_configuration()
        timing_v_1.append(time.time() - start_time)

        # version 2
        print("VERSION2")
        #start_config = config_space.sample_configuration(size=1)
        #sample_configs = config_space.sample_configuration(size=2)
        #sample_configs_values = [get_values(evaluation_config.get_dictionary(), variable_pipeline_steps) \
        #                         for evaluation_config in sample_configs]
        start_time = time.time()
        # new_configurations_2 = combine_configurations_batch_version2(config_space=config_space,
        #                                                              start_config=start_config,
        #                                                              complemented_configs_values=sample_configs_values,
        #                                                              constant_pipeline_steps=constant_pipeline_steps)
        for config in sample_configs:
            config.is_valid_configuration_vector()
        timing_v_2.append(time.time() - start_time)
        #print(len(new_configurations), len(new_configurations_2))

    print(np.mean(timing_v_1))
    print(np.mean(timing_v_2))



def run_experiment_sampling():
    pipeline_space = PipelineSpace()
    o_s = OneHotEncodingStep()
    i_s = ImputationStep()
    r_s = RescalingStep()
    b_s = BalancingStep()
    p_s = PreprocessingStep()
    c_s = ClassificationStep()
    pipeline_space.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s])
    constant_pipeline_steps = ["one_hot_encoder", "imputation", "rescaling",
                               "balancing", "feature_preprocessor"]
    variable_pipeline_steps = ["classifier"]

    cs_builder = ConfigSpaceBuilder(pipeline_space)
    config_space = cs_builder.build_config_space()

    timing_v_1 = []
    timing_v_2 = []
    for i in range(0, 5):
        print("Run: {}".format(i))
        # sample 1 start config

        # version 1
        start_time = time.time()
        sample_configs = config_space.sample_configuration(size=500)
        timing_v_1.append(time.time() - start_time)

        # version 2
        print("VERSION2")
        # start_config = config_space.sample_configuration(size=1)
        # sample_configs = config_space.sample_configuration(size=2)
        # sample_configs_values = [get_values(evaluation_config.get_dictionary(), variable_pipeline_steps) \
        #                         for evaluation_config in sample_configs]
        start_time = time.time()
        sample_configs_2 = config_space.sample_configuration_vector_checking(size=500)
        timing_v_2.append(time.time() - start_time)

        invalid_configs = []
        # for config in sample_configs_2:
        #     try:
        #         config.is_valid_configuration()
        #     except ValueError as v:
        #         exc_info = sys.exc_info()
        #         # Display the *original* exception
        #         traceback.print_exception(*exc_info)
        #         del exc_info
        #
        #         invalid_configs.append(config)
        #         print("Config not valid: {}".format(config))

        print("Nb of invalid configs: {}".format(len(invalid_configs)))
        print(len(sample_configs), len(sample_configs_2))

        print(np.mean(timing_v_1))
        print(np.mean(timing_v_2))

def run_experiment_forbidden():
    pipeline_space = PipelineSpace()
    o_s = OneHotEncodingStep()
    i_s = ImputationStep()
    r_s = RescalingStep()
    b_s = BalancingStep()
    p_s = PreprocessingStep()
    c_s = ClassificationStep()
    pipeline_space.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s])
    constant_pipeline_steps = ["one_hot_encoder", "imputation", "rescaling",
                               "balancing", "feature_preprocessor"]
    variable_pipeline_steps = ["classifier"]

    cs_builder = ConfigSpaceBuilder(pipeline_space)
    config_space = cs_builder.build_config_space()

    timing_v_1 = []
    timing_v_2 = []
    for i in range(0, 20):
        print("Run: {}".format(i))
        # sample 1 start config

        sample_configs = config_space.sample_configuration(size=500)

        # version 1
        start_time = time.time()
        for config in sample_configs:
            config_space._check_forbidden(config)
        timing_v_1.append(time.time() - start_time)

        # version 2
        print("VERSION2")


        start_time = time.time()
        for config in sample_configs:
            config_space._check_forbidden_vector(config.get_array())
        timing_v_2.append(time.time() - start_time)

    print(np.mean(timing_v_1))
    print(np.mean(timing_v_2))


def run_experiment():
    pipeline_space = PipelineSpace()
    o_s = OneHotEncodingStep()
    i_s = ImputationStep()
    r_s = RescalingStep()
    b_s = BalancingStep()
    p_s = PreprocessingStep()
    c_s = ClassificationStep()
    pipeline_space.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s])
    constant_pipeline_steps = ["one_hot_encoder", "imputation", "rescaling",
                               "balancing", "feature_preprocessor"]
    variable_pipeline_steps = ["classifier"]

    cs_builder = ConfigSpaceBuilder(pipeline_space)
    config_space = cs_builder.build_config_space()

    timing_v_1 = []
    timing_v_2 = []
    for i in range(0, 100):
        print("Run: {}".format(i))
        # sample 1 start config
        start_config = config_space.sample_configuration(size=1)

        # version 1
        start_time = time.time()
        sample_configs = config_space.sample_configuration(size=100)
        sample_configs_values = [get_values(evaluation_config.get_dictionary(), variable_pipeline_steps) \
                                 for evaluation_config in sample_configs]
        new_configurations = combine_configurations_batch_version1(config_space=config_space,
                                                                   start_config=start_config,
                                                                   complemented_configs_values=sample_configs_values,
                                                                   constant_pipeline_steps=constant_pipeline_steps)
        timing_v_1.append(time.time() - start_time)

        # version 2
        start_time = time.time()
        vector_values = get_vector_values(config_space, start_config, constant_pipeline_steps)
        sample_configs_2 = config_space.sample_configuration_with_given_values(size=100, given_values=vector_values)
        timing_v_2.append(time.time() - start_time)

    print(np.mean(timing_v_1))
    print(np.mean(timing_v_2))



    # # version 2
    # new_configurations_2 = combine_configurations_batch_version2(config_space=config_space,
    #                                                            start_config=start_config,
    #                                                            complemented_configs_values=sample_configs_values,
    #                                                            constant_pipeline_steps=constant_pipeline_steps)

    # print("Version 1 length: {}".format(len(new_configurations)))
    # print("Version 2 length: {}".format(len(new_configurations_2)))


def combine_configurations_batch_version1(config_space, start_config, complemented_configs_values, constant_pipeline_steps):
    constant_values = get_values(start_config.get_dictionary(), constant_pipeline_steps)
    batch = []
    for complemented_config_values in complemented_configs_values:
        new_config_values = {}
        new_config_values.update(constant_values)

        new_config_values.update(complemented_config_values)

        try:
            # start_time = time.time()
            config_object = Configuration(configuration_space=config_space,
                                          values=new_config_values)
            # print("Constructing configuration: {}".format(time.time() - start_time))
            batch.append(config_object)
        except ValueError as v:
            pass
    return batch

def combine_configurations_batch_version2(config_space, start_config, complemented_configs_values, constant_pipeline_steps):
    constant_values = get_values(start_config.get_dictionary(), constant_pipeline_steps)
    batch = []
    for complemented_config_values in complemented_configs_values:
        new_config_values = {}
        new_config_values.update(constant_values)

        new_config_values.update(complemented_config_values)

        try:
            # start_time = time.time()
            config_object = Configuration(configuration_space=config_space,
                                          values=new_config_values,
                                          vector_checking=False)
            # print("Constructing configuration: {}".format(time.time() - start_time))
            batch.append(config_object)
        except ValueError as v:
            pass
    return batch


def get_values(config_dict, pipeline_steps):
    value_dict = {}
    for step_name in pipeline_steps:
        for hp_name in config_dict:
            splt_hp_name = hp_name.split(":")
            if splt_hp_name[0] == step_name:
                value_dict[hp_name] = config_dict[hp_name]
    return value_dict

def get_vector_values(config_space, config, pipeline_steps):
    value_dict = {}
    for step_name in pipeline_steps:
        for hp_name in config.get_dictionary():
            splt_hp_name = hp_name.split(":")
            if splt_hp_name[0] == step_name:
                item_idx = config_space._hyperparameter_idx[hp_name]
                value_dict[hp_name] = config.get_array()[item_idx]
    return value_dict


if __name__ == "__main__":
    run_experiment_sampling()