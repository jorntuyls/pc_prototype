
import traceback
import sys

import numpy as np
import time

from sklearn.metrics import precision_score

from pc_smac.pc_smac.pipeline.pipeline_builder import PipelineBuilder

class PipelineRunner(object):

    def __init__(self, data, pipeline_space, runhistory, statistics, downsampling=None):
        # TODO Remove runhistory from arguments
        if downsampling:
            self.X_train = data["X_train"][:downsampling]
            self.y_train = data["y_train"][:downsampling]
        else:
            self.X_train = data["X_train"]
            self.y_train = data["y_train"]

        self.runtime_timing = {}
        self.runhistory = runhistory
        self.statistics = statistics
        self.pipeline_builder = PipelineBuilder(pipeline_space, caching=False, cache_directory=None)

    def run(self, config, instance, seed):
        """
            See ExecuteTARun class in SMAC3: https://github.com/automl/SMAC3
            Parameters
            ----------
                config : dictionary
                    dictionary param -> value
                instance : string
                    problem instance
                cutoff : double
                    runtime cutoff
                seed : int
                    random seed
                instance_specific: str
                    instance specific information (e.g., domain file or solution)

            Returns
            -------
                status: enum of StatusType (int)
                    {SUCCESS, TIMEOUT, CRASHED, ABORT}
                cost: float
                    cost/regret/quality (float) (None, if not returned by TA)
                runtime: float
                    runtime (None if not returned by TA)
                additional_info: dict
                    all further additional run information
        """

        print("start tae_runner")
        print(config, instance, seed)
        # start timer
        start_timer = time.time()

        self.runtime_timing = {}
        additional_info = {}

        pipeline = self.pipeline_builder.build_pipeline(config)

        # Cross validation as in scikit-learn:
        #   http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
        K = 3
        X_folds = np.array_split(self.X_train, K)
        y_folds = np.array_split(self.y_train, K)
        scores = list()
        try:
            for k in range(0,K):
                # Fit pipeline
                X_train = list(X_folds)
                X_valid = X_train.pop(k)
                X_train = np.concatenate(X_train)
                y_train = list(y_folds)
                y_valid = y_train.pop(k)
                y_train = np.concatenate(y_train)
                pipeline.fit(X_train, y_train)

                # Keep track of timing infomration
                self.add_runtime_timing(self.runtime_timing, pipeline.pipeline_info.get_timing_flat())
                print("TIMING: {}".format(pipeline.pipeline_info.get_timing()))

                # Validate pipeline
                y_pred = pipeline.predict(X_valid)
                prec_score = precision_score(y_valid, y_pred, average='macro')
                print("SCORE: {}".format(prec_score))
                scores.append(prec_score)
            cost = 1 - np.mean(scores)
        except ValueError as v:
            exc_info = sys.exc_info()
            cost = 1234567890
            # Display the *original* exception
            traceback.print_exception(*exc_info)
            del exc_info

        # Calculate score and total runtime
        runtime = time.time() - start_timer
        print("cost: {}, time: {}".format(cost, runtime))

        # Add information of this run to statistics
        run_information = {
            'cost': cost,
            'runtime': runtime,
            'pipeline_steps_timing': self.runtime_timing
        }
        self.statistics.add_run(config.get_dictionary(), run_information)

        print("stop tae_runner")
        return cost, additional_info

    def add_runtime_timing(self, dct, timing):
        for key in timing.keys():
            if key in dct.keys():
                dct[key] += timing[key]
            else:
                dct[key] = timing[key]

    def update_statistics(self, statistics):
        self.statistics = statistics





class CachedPipelineRunner(PipelineRunner):

    def __init__(self, data, pipeline_space, runhistory, statistics, cache_directory=None, downsampling=None):

        super(CachedPipelineRunner,self).__init__(data, pipeline_space, runhistory, statistics, downsampling=downsampling)

        self.pipeline_builder = PipelineBuilder(pipeline_space, caching=True, cache_directory=cache_directory)
        self.transformer_runtime_timing = {}
        self.cache_hits = {
            'total': 0,
            'cache_hits': 0
        }

    def run(self, config, instance, seed):

        print("start cached tae_runner")
        print(config, instance, seed)
        # start timer
        start_timer = time.time()

        self.runtime_timing = {}
        additional_info = {}

        pipeline = self.pipeline_builder.build_pipeline(config)

        # Cross validation as in scikit-learn:
        #   http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
        K = 3
        X_folds = np.array_split(self.X_train, K)
        y_folds = np.array_split(self.y_train, K)
        scores = list()
        try:
            for k in range(0, K):
                # Fit pipeline
                X_train = list(X_folds)
                X_valid = X_train.pop(k)
                X_train = np.concatenate(X_train)
                y_train = list(y_folds)
                y_valid = y_train.pop(k)
                y_train = np.concatenate(y_train)
                pipeline.fit(X_train, y_train)

                # Keep track of timing information
                self.add_runtime_timing(self.transformer_runtime_timing, pipeline.pipeline_info.get_preprocessor_timing())
                self.add_runtime_timing(self.runtime_timing, pipeline.pipeline_info.get_timing_flat())
                print("TIMING: {}".format(pipeline.pipeline_info.get_timing()))

                # Validate pipeline
                score_start = time.time()
                # TODO Does it make sense to cache the validation too? Or doesn't this take much time?
                y_pred = pipeline.predict(X_valid)
                prec_score = precision_score(y_valid, y_pred, average='macro')
                score_time = time.time() - score_start
                print("TIME: {}, SCORE: {}".format(score_time, prec_score))
                scores.append(prec_score)
            cost = 1 - np.mean(scores)
        except ValueError as v:
            exc_info = sys.exc_info()
            cost = 1234567890
            # Display the *original* exception
            traceback.print_exception(*exc_info)
            del exc_info

        # Update cache hits
        self.cache_hits['total'] += pipeline.pipeline_info.get_cache_hits()[0]
        self.cache_hits['cache_hits'] += pipeline.pipeline_info.get_cache_hits()[1]

        # Get reduction in runtime for cached configuration if it was not already cached
        # TODO insert this
        # if pipeline.pipeline_info.get_cache_hits()[1] == 0:
        t_rc = self._get_pipeline_steps_timing(self.transformer_runtime_timing, config)
        additional_info['t_rc'] = t_rc

        # Calculate score and total runtime
        runtime = time.time() - start_timer
        print("cost: {}, time: {}".format(cost, runtime))
        print("Total function evaluations: {}, cache hits: {}".format(self.cache_hits['total'],
                                                                      self.cache_hits['cache_hits']))

        # Add information of this run to statistics
        run_information = {
            'cost': cost,
            'runtime': runtime,
            'pipeline_steps_timing': self.runtime_timing,
            'cache_hits': self.cache_hits['cache_hits'],
            'total_evaluations': self.cache_hits['total']
        }
        self.statistics.add_run(config.get_dictionary(), run_information)

        print("stop cached tae_runner")
        return cost, additional_info

    def clean_cache(self):
        self.pipeline_builder.clean_cache()

    #### Private methods ####

    '''
    Return: List of tuples (dict, time) where dict is a cached algorithm (part of pipeline) configuration and
                time is runtime that this algorithm configuration took
    '''
    def _get_pipeline_steps_timing(self, timing, config):
        """

        Parameters
        ----------
        timing
        config

        Returns
        -------

        """
        t_rc = []
        for name in timing.keys():
            dict = {}
            splt_name = name.split(":")
            type = splt_name[0]
            algo_name = splt_name[1]
            for hp in config.keys():
                splt_hp = hp.split(":")
                if (splt_hp[0] == type and splt_hp[1] == '__choice__') \
                        or (splt_hp[0] == type and splt_hp[1] == algo_name):
                    dict[hp] = config[hp]
            t_rc.append((dict, timing[name]))

        return t_rc



class PipelineTester(object):

    def __init__(self, data, pipeline_space, downsampling=None):
        if downsampling:
            self.X_train = data["X_train"][:downsampling]
            self.y_train = data["y_train"][:downsampling]
            self.X_test = data["X_test"][:downsampling]
            self.y_test = data["y_test"][:downsampling]
        else:
            self.X_train = data["X_train"]
            self.y_train = data["y_train"]
            self.X_test = data["X_test"]
            self.y_test = data["y_test"]

        self.pipeline_builder = PipelineBuilder(pipeline_space, caching=False, cache_directory=None)

    def run(self, config):
        pipeline = self.pipeline_builder.build_pipeline(config)

        try:
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            score = precision_score(self.y_test, y_pred, average='macro')
        except ValueError as v:
            score = 0

        return score

    def get_performance(self, config):
        score = self.run(config)
        return 1 - score
