
import time
import numpy as np

from ConfigSpace.util import impute_inactive_values, get_one_exchange_neighbourhood
from smac.smbo.local_search import LocalSearch

class PCLocalSearch(LocalSearch):

    def __init__(self,
                 acquisition_function,
                 config_space,
                 epsilon=0.00001,
                 max_iterations=None,
                 rng=None):
        super(PCLocalSearch,self).__init__(acquisition_function,
                                           config_space,
                                           epsilon=epsilon,
                                           max_iterations=max_iterations,
                                           rng=rng)

    def maximize(self, start_point, cached_configs=[], *args):
        """
        Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters:
        ----------

        start_point:  np.array(1, D):
            The point from where the local search starts
        *args :
            Additional parameters that will be passed to the
            acquisition function

        Returns:
        -------

        incumbent np.array(1, D):
            The best found configuration
        acq_val_incumbent np.array(1,1) :
            The acquisition value of the incumbent

        """
        incumbent = start_point
        # Compute the acquisition value of the incumbent
        # First compute runtime discount of incumbent through caching of parts of the pipeline
        #   Do this before imputing inactive values!!
        incumbent_caching_discount = self._compute_caching_discounts([incumbent], cached_configs)
        #print("INCUMBENT CACHING DISCOUNT: {}".format(incumbent_caching_discount))
        incumbent_ = impute_inactive_values(incumbent)

        acq_val_incumbent = self.acquisition_function(
            incumbent_.get_array(), incumbent_caching_discount,
            *args)

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:

            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                self.logger.warn("Local search took already %d iterations." \
                                 "Is it maybe stuck in a infinite loop?", local_search_steps)

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            all_neighbors = get_one_exchange_neighbourhood(incumbent,
                                                           seed=self.rng.seed())
            self.rng.shuffle(all_neighbors)

            for neighbor in all_neighbors:
                s_time = time.time()

                # Compute runtime discount of neighbor through caching of parts of the pipeline
                neighbor_caching_discount = self._compute_caching_discounts([neighbor], cached_configs)
                #print("NEIGHBOR CACHING DISCOUNT: {}".format(neighbor_caching_discount))

                neighbor_ = impute_inactive_values(neighbor)

                n_array = neighbor_.get_array()

                acq_val = self.acquisition_function(n_array, neighbor_caching_discount, *args)

                neighbors_looked_at += 1

                time_n.append(time.time() - s_time)

                if acq_val > acq_val_incumbent + self.epsilon:
                    self.logger.debug("Switch to one of the neighbors")
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or (self.max_iterations != None
                                     and local_search_steps == self.max_iterations):
                self.logger.debug("Local search took %d steps and looked at %d configurations. "
                                  "Computing the acquisition value for one "
                                  "configuration took %f seconds on average.",
                                  local_search_steps, neighbors_looked_at, np.mean(time_n))
                break

        return incumbent, acq_val_incumbent

    def _compute_caching_discounts(self, configs, cached_configs):
        runtime_discounts = []
        for config in configs:
            discount = 0
            for cached_config in cached_configs:
                discount += self._caching_reduction(config, cached_config)
            runtime_discounts.append(discount)
        return np.array(runtime_discounts)

    def _caching_reduction(self, config, cached_config):
        '''

        Parameters
        ----------
        config
        cached_config: List of

        Returns
        -------

        '''
        r = [key for key in cached_config[0].keys() if config[key] != cached_config[0][key]]
        if r == []:
            return cached_config[1]
        return 0