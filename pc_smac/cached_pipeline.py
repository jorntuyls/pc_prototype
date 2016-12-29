

# This code is inspired by work done in the scikit-learn community
#   See PR #7990: https://github.com/scikit-learn/scikit-learn/pull/7990/

import time

from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.base import clone
from sklearn.externals.joblib import Memory
from sklearn.externals import six

class CachedPipeline(Pipeline):

    def __init__(self, steps, memory=Memory(cachedir=None, verbose=0)):
        self.memory = memory
        if isinstance(memory, six.string_types):
            self.memory = Memory(cachedir=memory, verbose=0)
        self.timing = {}
        super(CachedPipeline, self).__init__(steps)

    # Make own _fit method from Pipeline to incorporate caching an timing
    def _own_fit(self, X, y=None, **fit_params):
        #self._validate_steps()

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            start_time = time.time()
            print(name, transform)
            test = self.memory.cache(self._fit_single_transform)._get_argument_hash([transform, name, Xt,y],{})
            print("HASH FIT: {}".format(test))
            Xt = self.memory.cache(self._fit_single_transform)(transform, name, Xt,
                                y)
            print(Xt)
            self.timing[name] =  time.time() - start_time

        if self._final_estimator is None:
            return Xt, {}
        return Xt, fit_params_steps[self.steps[-1][0]]

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
            else:
                Xt = self._fit_single_transform(transform, name, idx_tr, Xt,
                                                               y, **fit_params_steps[name])
            self.timing[name] = time.time() - start_time

        if self._final_estimator is None:
            return Xt, {}
        return Xt, fit_params_steps[self.steps[-1][0]]

    def _test(self, *args, **kwargs):
        pass

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
            self.timing[self.steps[-1][0]] = time.time() - start_time
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
                #test = self.memory.cache(self._single_transform)._get_argument_hash([transform, Xt],{})
                #print("HASH TRANSFORM: {}".format(test))
                Xt = self._single_transform(transform, Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, y, **score_params)

    def _single_transform(self, transform, X):
        print("EVALUATE _SINGLE_TRANSFORM")
        return transform.transform(X)

    def _fit_single_transform(self, transform, name, idx_tr,  X, y, **fit_params_trans):
        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)

        print("EVALUATE _FIT_SINGLE_TRANSFORM")
        memory = self.memory
        clone_transformer = clone(transform)
        Xt, transform = memory.cache(_fit_transform_one)(
            clone_transformer, name,
            None, X, y,
            **fit_params_trans)
        self.steps[idx_tr] = (name, transform)

        return Xt



def _fit_transform_one(transformer, name, weight, X, y,
                               **fit_params):
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, transformer
    return res * weight, transformer
