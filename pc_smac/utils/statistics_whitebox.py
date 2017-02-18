
import time

from smac.stats.stats import Stats

"""
Class to hack the SMAC Stats class so we can change the logged time
"""
class WhiteboxStats(Stats):

    def __init__(self, scenario):
        super(WhiteboxStats, self).__init__(scenario)
        self.current_time = None
        self.scenario_copy = scenario

    def start_timing(self):
        super(WhiteboxStats, self).start_timing()
        self.current_time = time.time()

    def hack_time(self, time):
        self.current_time += time

    def get_used_wallclock_time(self):
        return self.current_time - self._start_time

    def get_remaing_time_budget(self):
        '''
            subtracts the runtime configuration budget with the used wallclock time
        '''
        if self.scenario_copy:
            return self.scenario_copy.wallclock_limit - self.get_used_wallclock_time()
        else:
            raise "Scenario is missing"

