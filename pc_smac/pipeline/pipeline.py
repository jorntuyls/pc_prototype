
import time

from sklearn.pipeline import Pipeline

from pc_smac.pc_smac.pipeline.cached_pipeline import PipelineInfo

from sklearn.externals import six

class OwnPipeline(Pipeline):

    def __init__(self, steps):
        super(OwnPipeline, self).__init__(steps)

        self.pipeline_info = PipelineInfo(caching=False)

    def _fit(self, X, y=None, **fit_params):
        # self._validate_steps()

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            start_time = time.time()

            if transform is None:
                pass
            elif hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params_steps[name])
            else:
                Xt = transform.fit(Xt, y, **fit_params_steps[name]) \
                    .transform(Xt)
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