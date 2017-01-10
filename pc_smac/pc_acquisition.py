

import numpy as np
from scipy.stats import norm

from smac.smbo.acquisition import EIPS

from ConfigSpace.configuration_space import Configuration


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
            #print("M RUNTIME BEFORE: {}".format(m_runtime))
            for i in range(0, len(m_runtime)):
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
        f = f / m_runtime
        f[s == 0.0] = 0.0

        if (f < 0).any():
            self.logger.error("Expected Improvement per Second is smaller than "
                              "0!")
            raise ValueError

        return f.reshape((-1, 1))

