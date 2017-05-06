
import traceback
import sys

import time

from pc_smac.pc_smac.pipeline.pipeline_builder import PipelineBuilder


class PipelineTimer(object):

    def __init__(self, data, pipeline_space, runhistory, statistics, downsampling=None):
        if downsampling:
            self.X_train = data["X_train"][:downsampling]
            self.y_train = data["y_train"][:downsampling]
        else:
            self.X_train = data["X_train"]
            self.y_train = data["y_train"]

        self.runtime_timing = {}
        self.runhistory = runhistory
        self.statistics = statistics
        self.pipeline_space = pipeline_space
        self.pipeline_builder = PipelineBuilder(pipeline_space, caching=False, cache_directory=None)

    def run(self, config, instance, seed):
        print("start pipeline timer")
        print(config, instance, seed)
        # start timer
        start_timer = time.time()

        self.runtime_timing = {}
        additional_info = {}

        pipeline = self.pipeline_builder.build_pipeline(config)
        try:
            # Fit pipeline
            pipeline.fit(self.X_train, self.y_train)

            # Keep track of timing infomration
            self.add_runtime_timing(self.runtime_timing, pipeline.pipeline_info.get_timing_flat())
            print("TIMING: {}".format(pipeline.pipeline_info.get_timing()))
            cost = 1
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
        self.statistics.add_run(config.get_dictionary(), run_information, config_origin=config.origin)

        print("stop pipeline timer")
        return cost, additional_info

    def add_runtime_timing(self, dct, timing):
        for key in timing.keys():
            if key in dct.keys():
                dct[key] += timing[key]
            else:
                dct[key] = timing[key]

    def update_statistics(self, statistics):
        self.statistics = statistics


class CachedPipelineTimer(PipelineTimer):

    def __init__(self, data, pipeline_space, runhistory, statistics, cache_directory=None, downsampling=None):
        super(CachedPipelineTimer, self).__init__(data, pipeline_space, runhistory, statistics,
                                                   downsampling=downsampling)

        self.pipeline_builder = PipelineBuilder(pipeline_space, caching=True, cache_directory=cache_directory)
        self.cached_transformer_runtime_timing = {}
        self.cache_hits = {
            'total': 0,
            'cache_hits': 0
        }

    def run(self, config, instance, seed):

        print("start cached pipeline timer")
        print(config, instance, seed)
        # start timer
        start_timer = time.time()

        self.runtime_timing = {}
        additional_info = {}

        pipeline = self.pipeline_builder.build_pipeline(config)

        try:
            # Fit pipeline
            pipeline.fit(self.X_train, self.y_train)

            # Keep track of timing information
            self.add_runtime_timing(self.cached_transformer_runtime_timing,
                                    pipeline.pipeline_info.get_cached_preprocessor_timing())
            self.add_runtime_timing(self.runtime_timing, pipeline.pipeline_info.get_timing_flat())
            print("TIMING: {}".format(pipeline.pipeline_info.get_timing()))

            cost = 1
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
        t_rc = self._get_pipeline_steps_timing(self.cached_transformer_runtime_timing, config)
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
        self.statistics.add_run(config.get_dictionary(), run_information, config_origin=config.origin)

        print("stop cached pipeline timer")
        return cost, additional_info

    def clean_cache(self):
        self.pipeline_builder.clean_cache()

    #### Private methods ####

    def _get_pipeline_steps_timing(self, timing, config):
        """

        Parameters
        ----------
        timing
        config

        Returns
        -------
        List of tuples (dict, time) where dict is a cached algorithm (part of pipeline) configuration and
            time is runtime that this algorithm configuration took

        """
        t_rc = []
        for name in timing.keys():
            dict = {}
            splt_name = name.split(":")
            type = splt_name[0]
            algo_name = splt_name[1]
            for hp in config.keys():
                splt_hp = hp.split(":")
                if self.pipeline_space.is_step_infront_of_step(splt_hp[0], type):
                    dict[hp] = config[hp]
                if (splt_hp[0] == type and splt_hp[1] == '__choice__') \
                        or (splt_hp[0] == type and splt_hp[1] == algo_name):
                    dict[hp] = config[hp]
            t_rc.append((dict, timing[name]))
        return t_rc


