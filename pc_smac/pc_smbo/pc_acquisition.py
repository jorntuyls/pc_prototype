
import time
import numpy as np
from scipy.stats import norm

from smac.smbo.acquisition import EIPS

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
import ConfigSpace.util

class PCAquisitionFunction(object):

    def __init__(self, acquisition_func, config_space, runhistory, constant_pipeline_steps, variable_pipeline_steps):
        self.acquisition_func = acquisition_func
        self.config_space = config_space
        self.runhistory = runhistory
        self.constant_pipeline_steps = constant_pipeline_steps
        self.variable_pipeline_steps = variable_pipeline_steps

    def __call__(self, configs):
        # TODO !! EI
        caching_discounts = self._compute_caching_discounts(configs, self.runhistory.get_cached_configurations())
        imputed_configs = map(ConfigSpace.util.impute_inactive_values,
                              configs)
        imputed_configs = [x.get_array()
                           for x in imputed_configs]
        imputed_configs = np.array(imputed_configs,
                                   dtype=np.float64)
        return self.acquisition_func(imputed_configs)

    def update(self, **kwargs):
        self.acquisition_func.update(**kwargs)

    def marginalized_prediction(self, configs, evaluation_configs=None):
        start_time = time.time()
        evaluation_configs_values = [self._get_values(evaluation_config.get_dictionary(), self.variable_pipeline_steps) \
                                     for evaluation_config in evaluation_configs]
        marg_acq_values = [self.get_marginalized_acquisition_value(config=config, evaluation_configs_values=evaluation_configs_values) for config in configs]
        # print("MARG ACQUISITION VALUES: {}".format(marg_acq_values))
        print("Compute marginalized acquisition values: {}".format(time.time() - start_time))

        return np.array(marg_acq_values, dtype=np.float64)

    #### HELPER FUNCTIONS ####
    def get_marginalized_acquisition_value(self, config, evaluation_configs_values=None, num_points=100):
        start_time = time.time()
        sample_configs = self._combine_configurations_batch(config, evaluation_configs_values) if evaluation_configs_values \
            else [self._get_variant_config(start_config=config) for i in range(0, num_points)]
        print("List construction: {}".format(time.time() - start_time))

        start_time = time.time()
        caching_discounts = self._compute_caching_discounts(sample_configs,
                                                            self.runhistory.get_cached_configurations())
        print("Compute caching discounts: {}".format(time.time() - start_time))

        start_time= time.time()
        imputed_configs = map(ConfigSpace.util.impute_inactive_values,
                              sample_configs)
        imputed_configs = [x.get_array()
                           for x in imputed_configs]
        imputed_configs = np.array(imputed_configs,
                                   dtype=np.float64)
        print("Compute imputed configs: {}".format(time.time() - start_time))

        #acq_values = self.acquisition_func(imputed_configs, caching_discounts)
        start_time = time.time()
        acq_values = self.acquisition_func(imputed_configs)
        print("Acquisition function evaluation: {}".format(time.time() - start_time))
        return np.mean(acq_values)

    def _get_variant_config(self, start_config, origin=None):
        next_config = start_config
        i = 0
        while i < 1000:
            try:
                start_time = time.time()
                sample_config = self.config_space.sample_configuration()
                #print("Sample config: {}".format(time.time() - start_time))
                start_time = time.time()
                next_config = self._combine_configurations(start_config, sample_config)
                #print("Combine config: {}, {}".format(time.time() - start_time, i))
                next_config.origin=origin
                break
            except ValueError as v:
                i += 1
        # TODO hack for now to combine preprocessing part of one configuration with classification part of all the others
        return next_config

    def _combine_configurations(self, start_config, complemented_config):
        constant_values = self._get_values(start_config.get_dictionary(), self.constant_pipeline_steps)
        new_config_values = {}
        new_config_values.update(constant_values)

        variable_values = self._get_values(complemented_config.get_dictionary(), self.variable_pipeline_steps)
        new_config_values.update(variable_values)

        return Configuration(configuration_space=self.config_space,
                             values=new_config_values)

    def _combine_configurations_batch(self, start_config, complemented_configs_values):
        constant_values = self._get_values(start_config.get_dictionary(), self.constant_pipeline_steps)
        batch = []
        for complemented_config_values in complemented_configs_values:
            new_config_values = {}
            new_config_values.update(constant_values)

            new_config_values.update(complemented_config_values)

            try:
                #start_time = time.time()
                config_object = Configuration(configuration_space=self.config_space,
                                              values=new_config_values)
                #print("Constructing configuration: {}".format(time.time() - start_time))
                batch.append(config_object)
            except ValueError as v:
                pass
        return batch

    def _get_values(self, config_dict, pipeline_steps):
        value_dict = {}
        for step_name in pipeline_steps:
            for hp_name in config_dict:
                splt_hp_name = hp_name.split(":")
                if splt_hp_name[0] == step_name:
                    value_dict[hp_name] = config_dict[hp_name]
        return value_dict

    # TODO the compute caching discounts methods are located in two places: in the select configuration procedure and also in the local search
    def _compute_caching_discounts(self, configs, cached_configs):
        runtime_discounts = []
        for config in configs:
            discount = 0
            for cached_config in cached_configs:
                discount += self._caching_reduction(config, cached_config)
            runtime_discounts.append(discount)
        return runtime_discounts

    def _caching_reduction(self, config, cached_config):
        '''

        Parameters
        ----------
        config:         the new configuration
        cached_config:  the cached configuration

        Returns
        -------
            The runtime discount for this configuration, given the cached configuration if there is one, otherwise 0
        '''
        r = [key for key in cached_config[0].keys() if config[key] != cached_config[0][key]]
        # print("_caching_reduction: {}".format(r))
        if r == []:
            return cached_config[1]
        return 0

class PCAquisitionFunctionWithCachingReduction(PCAquisitionFunction):

    def __init__(self, acquisition_func, config_space, runhistory, constant_pipeline_steps, variable_pipeline_steps):
        self.acquisition_func = acquisition_func
        self.config_space = config_space
        self.runhistory = runhistory
        self.constant_pipeline_steps = constant_pipeline_steps
        self.variable_pipeline_steps = variable_pipeline_steps

    def __call__(self, configs):
        # TODO !! EI
        caching_discounts = self._compute_caching_discounts(configs, self.runhistory.get_cached_configurations())
        imputed_configs = map(ConfigSpace.util.impute_inactive_values,
                              configs)
        imputed_configs = [x.get_array()
                           for x in imputed_configs]
        imputed_configs = np.array(imputed_configs,
                                   dtype=np.float64)
        return self.acquisition_func(imputed_configs, caching_discounts)

    def get_marginalized_acquisition_value(self, config, evaluation_configs_values=None, num_points=100):
        start_time = time.time()
        sample_configs = self._combine_configurations_batch(config,
                                                            evaluation_configs_values) if evaluation_configs_values \
            else [self._get_variant_config(start_config=config) for i in range(0, num_points)]
        print("List construction: {}".format(time.time() - start_time))

        start_time = time.time()
        caching_discounts = self._compute_caching_discounts(sample_configs,
                                                            self.runhistory.get_cached_configurations())
        print("Compute caching discounts: {}".format(time.time() - start_time))

        start_time = time.time()
        imputed_configs = map(ConfigSpace.util.impute_inactive_values,
                              sample_configs)
        imputed_configs = [x.get_array()
                           for x in imputed_configs]
        imputed_configs = np.array(imputed_configs,
                                   dtype=np.float64)
        print("Compute imputed configs: {}".format(time.time() - start_time))

        # acq_values = self.acquisition_func(imputed_configs, caching_discounts)
        start_time = time.time()
        acq_values = self.acquisition_func(imputed_configs, caching_discounts)
        print("Acquisition function evaluation: {}".format(time.time() - start_time))
        return np.mean(acq_values)





class PCEIPS(EIPS):

    def __init__(self,
                 model,
                 par=0.01,
                 **kwargs):

        super(PCEIPS, self).__init__(model)

    def _compute(self, X, runtime_discount=None, derivative=False, **kwargs):
        """
        Computes the EIPS value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            Raises NotImplementedError if True.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement per Second of X
        """

        if derivative:
            raise NotImplementedError()

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        assert m.shape[1] == 2
        assert v.shape[1] == 2
        m_cost = m[:, 0]
        v_cost = v[:, 0]
        # The model already predicts np.log(runtime)
        m_runtime = m[:, 1]
        s = np.sqrt(v_cost)
        # Take into account runtime discount for cached configuration parts
        if runtime_discount:
            for i in range(0, len(m_runtime)):
                # runhistory2epm4eips returns 'np.log(1 + run.time)'as runtime cost
                if np.exp(m_runtime[i]) - runtime_discount[i] < 1:
                    print(np.exp(m_runtime[i]), runtime_discount[i])
                m_runtime[i] =\
                    np.log(np.exp(m_runtime[i]) - runtime_discount[i]) if np.exp(m_runtime[i]) - runtime_discount[i] > 1 else m_runtime[i]

            #print("M RUNTIME AFTER: {}".format(m_runtime))


        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        z = (self.eta - m_cost - self.par) / s
        f = (self.eta - m_cost - self.par) * norm.cdf(z) + s * norm.pdf(z)
        # TODO !! EI
        f = f / m_runtime
        f[s == 0.0] = 0.0

        if (f < 0).any():
            self.logger.error("Expected Improvement per Second is smaller than "
                              "0!")
            raise ValueError

        return f.reshape((-1, 1))

