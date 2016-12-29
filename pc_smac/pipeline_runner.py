
import warnings

import numpy as np
import time
import sys

from smac.tae.execute_ta_run import StatusType
from smac.tae.execute_ta_run import ExecuteTARun

from pipeline_builder import PipelineBuilder

class PipelineRunner(ExecuteTARun):

    def __init__(self, data, pipeline_space, runhistory, downsampling=None):
        if downsampling:
            self.X_train = data["X_train"][:downsampling]
            self.y_train = data["y_train"][:downsampling]
        else:
            self.X_train = data["X_train"]
            self.y_train = data["y_train"]


        self.pipeline_space = pipeline_space
        self.runhistory = runhistory
        self.pipeline_builder = PipelineBuilder(pipeline_space)

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
        additional_info = {}
        start_timer = time.time()

        pipeline = self.pipeline_builder.build_pipeline(config)

        status = StatusType.SUCCESS
        score = 0

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
                print("TIMING: {}".format(pipeline.timing))
                score = pipeline.score(X_valid, y_valid)
                print(score)
                scores.append(score)
            score = np.mean(scores)
        except ValueError as v:
            print("ValueError: {}".format(v))
            status = StatusType.CRASHED
            score = 0

        cost = 1 - score
        print("cost: {}".format(cost))
        runtime = time.time() - start_timer
        print("time: {}".format(runtime))
        print("Status: {}".format(status))

        #TODO update runhistory
        self.runhistory.add(config=config,
                            cost=cost, time=runtime, status=status,
                            instance_id=instance, seed=seed,
                            additional_info=additional_info)

        print("stop tae_runner")
        return status, cost, runtime, {}

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
