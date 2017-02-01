
import copy
import numpy as np
from itertools import product

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

import pc_smac.pc_smac.config_space.create_search_space_util as utility
from pc_smac.pc_smac.utils.constants import *


class ConfigSpaceBuilder:

    def __init__(self, pipeline_space):
        self.pipeline_space = pipeline_space

    def build_config_space(self, seed=None, dataset_properties=None):
        cs = ConfigurationSpace() if seed == None else ConfigurationSpace(seed=seed)
        pipeline = self.pipeline_space.get_pipeline_steps_names_and_objects()
        vanilla_cs = self._get_hyperparameter_search_space(cs, {}, exclude=None, include=None,
                                                   pipeline=pipeline)
        cs = self.add_forbidden_clauses(vanilla_cs, pipeline, dataset_properties)
        print(cs)

        return cs

    # def build_vanilla_config_space(self, cs):
    #     for ps in self.pipeline_space.get_pipeline_steps():
    #         cs.add_hyperparameter(CategoricalHyperparameter(ps.get_name(),
    #                                                         ps.get_node_names()))
    #         for node in ps.get_nodes():
    #             sub_cs = node.get_hyperparameter_search_space()
    #             cs.add_configuration_space(node.get_name(), sub_cs)
    #     return cs

    def _get_hyperparameter_search_space_pipeline_step(self, ps, include=None):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter('__choice__',
                                                        ps.get_node_names()))

        for node in ps.get_nodes():
            if include is not None and node not in include:
                continue
            sub_cs = node.get_hyperparameter_search_space()
            cs.add_configuration_space(node.get_name(), sub_cs)
        return cs

    def _get_hyperparameter_search_space(self, cs, dataset_properties, exclude,
                                             include, pipeline):
        if include is None:
            include = {}

        keys = [pair[0] for pair in pipeline]
        for key in include:
            if key not in keys:
                raise ValueError('Invalid key in include: %s; should be one '
                                 'of %s' % (key, keys))

        if exclude is None:
            exclude = {}

        keys = [pair[0] for pair in pipeline]
        for key in exclude:
            if key not in keys:
                raise ValueError('Invalid key in exclude: %s; should be one '
                                 'of %s' % (key, keys))

        if 'sparse' not in dataset_properties:
            # This dataset is probaby dense
            dataset_properties['sparse'] = False
        if 'signed' not in dataset_properties:
            # This dataset probably contains unsigned data
            dataset_properties['signed'] = False

        matches = utility.get_match_array(
            pipeline, dataset_properties, include=include, exclude=exclude)

        # Now we have only legal combinations at this step of the pipeline
        # Simple sanity checks
        assert np.sum(matches) != 0, "No valid pipeline found."

        assert np.sum(matches) <= np.size(matches), \
            "'matches' is not binary; %s <= %d, %s" % \
            (str(np.sum(matches)), np.size(matches), str(matches.shape))

        # Iterate each dimension of the matches array (each step of the
        # pipeline) to see if we can add a hyperparameter for that step
        for node_idx, n_ in enumerate(pipeline):
            node_name, node = n_

            is_choice = node.get_nb_nodes() > 1

            # if the node isn't a choice we can add it immediately because it
            #  must be active (if it wouldn't, np.sum(matches) would be zero
            if not is_choice:
                sub_cs = self._get_hyperparameter_search_space_pipeline_step(node)
                cs.add_configuration_space(node_name, sub_cs)
            # If the node isn't a choice, we have to figure out which of it's
            #  choices are actually legal choices
            else:
                choices_list = utility. \
                    find_active_choices(matches, node, node_idx,
                                        dataset_properties,
                                        include.get(node_name),
                                        exclude.get(node_name))
                sub_cs = self._get_hyperparameter_search_space_pipeline_step(node, include=choices_list)
                cs.add_configuration_space(node_name, sub_cs)

        # And now add forbidden parameter configurations
        # According to matches
        if np.sum(matches) < np.size(matches):
            cs = utility.add_forbidden(
                conf_space=cs, pipeline=pipeline, matches=matches,
                dataset_properties=dataset_properties, include=include,
                exclude=exclude)

        return cs

    def add_forbidden_clauses(self, cs, pipeline, dataset_properties):
        classifiers = cs.get_hyperparameter('classifier:__choice__').choices
        preprocessors = cs.get_hyperparameter('feature_preprocessor:__choice__').choices
        available_classifiers = pipeline[-1][1]
        available_preprocessors = pipeline[-2][1]

        possible_default_classifier = copy.copy([node.get_name() for node in available_classifiers.get_nodes()])
        default = cs.get_hyperparameter('classifier:__choice__').default
        del possible_default_classifier[possible_default_classifier.index(default)]

        # A classifier which can handle sparse data after the densifier is
        # forbidden for memory issues
        for name in classifiers:
            if SPARSE in available_classifiers.get_node(name).get_properties()['input']:
                if 'densifier' in preprocessors:
                    while True:
                        try:
                            cs.add_forbidden_clause(
                                ForbiddenAndConjunction(
                                    ForbiddenEqualsClause(
                                        cs.get_hyperparameter(
                                            'classifier:__choice__'), name),
                                    ForbiddenEqualsClause(
                                        cs.get_hyperparameter(
                                            'feature_preprocessor:__choice__'), 'densifier')
                                ))
                            # Success
                            break
                        except ValueError:
                            # Change the default and try again
                            try:
                                default = possible_default_classifier.pop()
                            except IndexError:
                                raise ValueError("Cannot find a legal default configuration.")
                            cs.get_hyperparameter(
                                'classifier:__choice__').default = default

        # which would take too long
        # Combinations of non-linear models with feature learning:
        classifiers_ = ["adaboost", "decision_tree", "extra_trees",
                        "gradient_boosting", "k_nearest_neighbors",
                        "libsvm_svc", "random_forest", "gaussian_nb",
                        "decision_tree", "xgradient_boosting"]
        feature_learning = ["kitchen_sinks", "nystroem_sampler"]

        for c, f in product(classifiers_, feature_learning):
            if c not in classifiers:
                continue
            if f not in preprocessors:
                continue
            while True:
                try:
                    cs.add_forbidden_clause(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "classifier:__choice__"), c),
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "preprocessor:__choice__"), f)))
                    break
                except KeyError:
                    break
                except ValueError as e:
                    # Change the default and try again
                    try:
                        default = possible_default_classifier.pop()
                    except IndexError:
                        raise ValueError(
                            "Cannot find a legal default configuration.")
                    cs.get_hyperparameter(
                        'classifier:__choice__').default = default

        # Won't work
        # Multinomial NB etc don't use with features learning, pca etc
        classifiers_ = ["multinomial_nb"]
        preproc_with_negative_X = ["kitchen_sinks", "pca", "truncatedSVD",
                                   "fast_ica", "kernel_pca", "nystroem_sampler"]

        for c, f in product(classifiers_, preproc_with_negative_X):
            if c not in classifiers:
                continue
            if f not in preprocessors:
                continue
            while True:
                try:
                    cs.add_forbidden_clause(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "preprocessor:__choice__"), f),
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "classifier:__choice__"), c)))
                    break
                except KeyError:
                    break
                except ValueError:
                    # Change the default and try again
                    try:
                        default = possible_default_classifier.pop()
                    except IndexError:
                        raise ValueError(
                            "Cannot find a legal default configuration.")
                    cs.get_hyperparameter(
                        'classifier:__choice__').default = default
        return cs
