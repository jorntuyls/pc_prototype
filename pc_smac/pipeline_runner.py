
import traceback
import sys

import numpy as np
import time

from smac.tae.execute_ta_run import StatusType
from smac.tae.execute_ta_run import ExecuteTARun

from sklearn.metrics import precision_score, accuracy_score, confusion_matrix

from pipeline_builder import PipelineBuilder

class PipelineRunner(ExecuteTARun):

    def __init__(self, data, pipeline_space, runhistory, cache_directory=None, downsampling=None):
        if downsampling:
            self.X_train = data["X_train"][:downsampling]
            self.y_train = data["y_train"][:downsampling]
        else:
            self.X_train = data["X_train"]
            self.y_train = data["y_train"]


        self.pipeline_space = pipeline_space
        self.runhistory = runhistory
        self.pipeline_builder = PipelineBuilder(pipeline_space, cache_directory)

        self.runtime_timing = {}
        self.cache_hits = {
            'total': 0,
            'cache_hits': 0
        }

    def start(self, config,
                    instance=None,
                    seed=None,
                    cutoff=np.inf,
                    instance_specific=None):
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
        print(config, instance, cutoff, seed, instance_specific)
        # start timer
        start_timer = time.time()

        self.runtime_timing = {}
        additional_info = {}
        status = StatusType.SUCCESS
        score = 0

        pipeline = self.pipeline_builder.build_pipeline(config)

        # Cross validation as in scikit-learn:
        #   http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
        K = 3
        X_folds = np.array_split(self.X_train, K)
        y_folds = np.array_split(self.y_train, K)
        scores = list()
        try:
            for k in range(0,K):
                X_train = list(X_folds)
                X_valid = X_train.pop(k)
                X_train = np.concatenate(X_train)
                y_train = list(y_folds)
                y_valid = y_train.pop(k)
                y_train = np.concatenate(y_train)
                pipeline.fit(X_train, y_train)
                self._add_runtime_timing(pipeline.pipeline_info.get_preprocessor_timing())
                print("TIMING: {}".format(pipeline.pipeline_info.get_timing()))
                score_start = time.time()
                #TODO Does it make sense to cache the validation too? Or doesn't this take much time?
                yt = pipeline.predict(X_valid)
                prec_score = precision_score(y_valid, yt, average='macro')
                score_time = time.time() - score_start
                print("TIME: {}, SCORE: {}".format(score_time, prec_score))
                scores.append(prec_score)
            score = np.mean(scores)
        except ValueError as v:
            exc_info = sys.exc_info()
            status = StatusType.CRASHED
            score = 0
            # Display the *original* exception
            traceback.print_exception(*exc_info)
            del exc_info

        # Get reduction in runtime for cached configuration
        t_rc = self._get_pipeline_steps_timing(self.runtime_timing, config)
        additional_info['t_rc'] = t_rc

        # Update cache hits
        self.cache_hits['total'] += pipeline.pipeline_info.get_cache_hits()[0]
        self.cache_hits['cache_hits'] += pipeline.pipeline_info.get_cache_hits()[1]

        # Calculate score and total runtime
        cost = 1 - score
        runtime = time.time() - start_timer
        print("cost: {}, time: {}, status: {}".format(cost, runtime, status))
        print("Total function evaluations: {}, cache hits: {}".format(self.cache_hits['total'],
                                                                      self.cache_hits['cache_hits']))

        # Update runhistory
        self.runhistory.add(config=config,
                            cost=cost, time=runtime, status=status,
                            instance_id=instance, seed=seed,
                            additional_info=additional_info)

        print("stop tae_runner")
        return status, cost, runtime, additional_info

    #### Private methods ####

    def _add_runtime_timing(self, timing):
        for key in timing.keys():
            if key in self.runtime_timing.keys():
                self.runtime_timing[key] += timing[key]
            else:
                self.runtime_timing[key] = timing[key]

    def _get_pipeline_steps_timing(self, timing, config):
        t_rc = []
        for name in timing.keys():
            dict = {}
            splt_name = name.split(":")
            type = splt_name[0]
            algo_name = splt_name[1]
            for hp in config.keys():
                splt_hp = hp.split(":")
                if hp == type or (len(splt_hp) == 2 and splt_hp[0] == algo_name):
                    dict[hp] = config[hp]
            t_rc.append((dict, timing[name]))

        return t_rc


class PipelineTester(object):

    def __init__(self, data, pipeline_space, metric=None):
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]
        self.metric = metric
        self.pipeline_space = pipeline_space
        self.pipeline_builder = PipelineBuilder(pipeline_space)

    def run(self, config):
        pipeline = self.pipeline_builder.build_pipeline(config)

        score = 1
        if self.metric:
            pipeline.fit(self.X_train, self.y_train)
            prediction = pipeline.predict(self.X_test)
            score = self.metric.calculate_score(self.y_test, prediction)
        else:
            score = pipeline.score(self.X_test, self.y_test)

        return score
