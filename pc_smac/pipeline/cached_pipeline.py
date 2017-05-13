

# This code is inspired by work done in the scikit-learn community
#   See PR #7990: https://github.com/scikit-learn/scikit-learn/pull/7990/

import time
import shutil
import numpy as np

from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.base import clone
from sklearn.externals.joblib import Memory
from sklearn.externals import six

# Use global variables to calculate the number of cache hits
FIT_SINGLE_TRANSFORM_EVALUATIONS = 0
FIT_TRANSFORM_ONE_EVALUATIONS = 0

class CachedPipeline(Pipeline):

    def __init__(self, steps, cached_step_names, memory=Memory(cachedir=None, verbose=0), min_runtime_for_caching=1, run_instance=None):
        self.memory = memory
        if isinstance(memory, six.string_types):
            self.memory = Memory(cachedir=memory, verbose=0)

        self.pipeline_info = PipelineInfo(caching=True)
        global FIT_SINGLE_TRANSFORM_EVALUATIONS
        FIT_SINGLE_TRANSFORM_EVALUATIONS = 0
        global FIT_TRANSFORM_ONE_EVALUATIONS
        FIT_TRANSFORM_ONE_EVALUATIONS = 0

        self.cached_step_names = cached_step_names
        self.min_runtime_for_caching = min_runtime_for_caching

        self.run_instance = run_instance

        super(CachedPipeline, self).__init__(steps)

    def _fit(self, X, y=None, **fit_params):
        # self._validate_steps()

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for idx_tr, (name, transform) in enumerate(self.steps[:-1]):
            start_time = time.time()
            if transform is None:
                pass
            elif name in self.cached_step_names:
                #hash_Xt = hash(str(Xt))
                #hash_Xt = 1
                previous_transformers = self.steps[:idx_tr]
                print(previous_transformers, self.run_instance)
                Xt, output_dir = self._fit_single_transform_cached(transform, name, previous_transformers, self.run_instance, idx_tr, Xt,
                                                                    y, **fit_params_steps[name])
                timing = time.time() - start_time
                # TODO Timing > 1
                if timing > self.min_runtime_for_caching or self.pipeline_info.get_cache_hits()[1] == self.pipeline_info.get_cache_hits()[0]:
                    self.pipeline_info.add_cached_preprocessor_timing(name, timing)
                    print("Cache output directory: {}, timing: {}".format(output_dir, timing))
                else:
                    print("Remove output directory: {}, timing: {}".format(output_dir, timing))
                    start_time = time.time()
                    shutil.rmtree(output_dir, ignore_errors=True)
                    print("Time to remove cache: {}".format(time.time() - start_time))
            else:
                Xt = self._fit_single_transform(transform, name, None, Xt, y, **fit_params_steps[name])
            self.pipeline_info.add_preprocessor_timing(name, time.time() - start_time)

        if self._final_estimator is None:
            return Xt, {}
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        Xt, fit_params = self._fit(X, y, **fit_params)
        if self._final_estimator is not None:
            start_time = time.time()
            self._final_estimator.fit(Xt, y, **fit_params)
            self.pipeline_info.add_estimator_timing(self.steps[-1][0], time.time() - start_time)
        return self

    def score(self, X, y, sample_weight=None):
        """Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = self._single_transform(transform, Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, y, **score_params)

    def _single_transform(self, transform, X):
        print("EVALUATE _SINGLE_TRANSFORM")
        return transform.transform(X)

    def _fit_single_transform(self, transformer, name, weight, X, y, **fit_params):
        if hasattr(transformer, 'fit_transform'):
            res = transformer.fit_transform(X, y, **fit_params)
        else:
            res = transformer.fit(X, y, **fit_params).transform(X)
        # if we have a weight for this transformer, multiply output
        if weight is None:
            return res
        return res * weight

    def _fit_single_transform_cached(self, transform, name, previous_transformers, run_instance, idx_tr,  X, y, **fit_params_trans):

        #print("EVALUATE _FIT_SINGLE_TRANSFORM")
        global FIT_SINGLE_TRANSFORM_EVALUATIONS
        FIT_SINGLE_TRANSFORM_EVALUATIONS += 1

        memory = self.memory
        clone_transformer = clone(transform)
        fit_tranform_one_cached = memory.cache(_fit_transform_one, ignore=["X", "y"])
        Xt, new_transform = fit_tranform_one_cached(
            clone_transformer, name, previous_transformers, run_instance,
            None, X, y,
            **fit_params_trans)
        output_dir, _ = fit_tranform_one_cached._get_output_dir(
            transform, name, previous_transformers, run_instance,
            None, X, y,
            **fit_params_trans)
        self.steps[idx_tr] = (name, new_transform)

        #print("END EVALUATE _FIT_SINGLE_TRANSFORM")

        return Xt, output_dir

def _fit_transform_one(transformer, name, previous_transformers, run_instance, weight, X, y,
                               **fit_params):
    global FIT_TRANSFORM_ONE_EVALUATIONS
    FIT_TRANSFORM_ONE_EVALUATIONS += 1
    #print("NO CACHE HIT")
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    res = res.astype(np.float32)
    if weight is None:
        return res, transformer
    return res * weight, transformer


class PipelineInfo(object):

    def __init__(self, caching):
        self.caching = caching
        self.timing = {
            'preprocessors': {},
            'cached_preprocessors': {},
            'estimators': {}
        }
        self.cache_hits = 0

    def add_preprocessor_timing(self, name, runtime):
        self.timing['preprocessors'][name] = runtime

    def add_cached_preprocessor_timing(self, name, runtime):
        self.timing['cached_preprocessors'][name] = runtime

    def get_preprocessor_timing(self):
        return self.timing['preprocessors']

    def get_cached_preprocessor_timing(self):
        return self.timing['cached_preprocessors']

    def add_estimator_timing(self, name, runtime):
        self.timing['estimators'][name] = runtime

    def get_estimator_timing(self):
        return self.timing['estimators']

    def get_timing(self):
        return self.timing

    def get_timing_flat(self):
        dct = self.get_preprocessor_timing().copy()
        dct.update(self.get_estimator_timing())
        return dct

    def get_cache_hits(self):
        if self.caching == True:
            global FIT_TRANSFORM_ONE_EVALUATIONS, FIT_SINGLE_TRANSFORM_EVALUATIONS
            return  (FIT_SINGLE_TRANSFORM_EVALUATIONS, FIT_SINGLE_TRANSFORM_EVALUATIONS - FIT_TRANSFORM_ONE_EVALUATIONS)
        else:
            return 0


