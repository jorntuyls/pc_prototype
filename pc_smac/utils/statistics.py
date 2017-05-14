
import os
import time
import json

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter


class Statistics(object):

    def __init__(self, stamp, output_dir, information: dict, total_runtime=None, time_precision=None):
        self.stamp = stamp
        self.output_dir = self._set_output_dir(output_dir)
        self.stat_information = information
        self.total_runtime = total_runtime
        self.time_precision = time_precision

        self.runs = []
        self.incumbents = []

        self.start_time = None

        # Output files, incumbent and info files not used for now
        if self.output_dir[-1] == "/":
            self.run_file = self.output_dir + "statistics_" + str(self.stamp) + "_runs.json"
            self.inc_file = self.output_dir + "statistics_" + str(self.stamp) + "_incumbents.json"
            self.inc_log_file = self.output_dir + "statistics_logging_" + str(self.stamp) + "_incumbents.json"
        else:
            self.run_file = self.output_dir + "/statistics_" + str(self.stamp) + "_runs.json"
            self.inc_file = self.output_dir + "/statistics_" + str(self.stamp) + "_incumbents.json"
            self.inc_log_file = self.output_dir + "/statistics_logging_" + str(self.stamp) + "_incumbents.json"

        #self.info_file = self.output_dir + "statistics_info_" + str(self.stamp) + ".json"

    def start_timer(self):
        self.start_time = time.time()
        return self.start_time

    def get_time_point(self):
        if self.start_time == None:
            raise ValueError("Timer is not yet started!")
        return time.time() - self.start_time

    def add_run(self, config: dict, information: dict, config_origin="Unknown"):
        time_point = self.get_time_point()
        run = self._add_run(self.runs, config, time_point, information, config_origin)
        # Append run directly to json file
        self._save_json([run], self.run_file)
        return time_point

    def get_run_trajectory(self):
        return self.runs

    def add_new_incumbent(self, incumbent: dict, information: dict, config_origin="Unknown"):
        time_point = self.get_time_point()
        inc_run = self._add_incumbent(self.incumbents, incumbent, time_point, information, config_origin)
        # Append run directly to json logging file
        self._save_json([inc_run], self.inc_log_file)
        return time_point

    def add_incumbents_trajectory(self, trajectory):
        self.incumbents = trajectory
        # Add to real incumbent file
        self._open_file(self.inc_file)
        self._save_json(self.incumbents, self.inc_file)

    def get_incumbent_trajectory(self, config_space=None):
        if config_space == None:
            raise ValueError("config space should not be none")
        if self.incumbents != []:
            for line in self.incumbents:
                line['incumbent'] = self._convert_dict_to_config(line['incumbent'], config_space=config_space)
            return self.incumbents
        else:
            return self._read_incumbent_file(self.inc_log_file, config_space)

    def get_incumbent_trajectory_dicts(self):
        if self.incumbents != []:
            return self.incumbents
        else:
            return self._read_json_file(self.inc_log_file)

    def save(self):
        # Create and clean files
        self._clean_files([self.run_file, self.inc_file])
        # Save info to files
        self._save_json(self.runs, self.run_file)
        self._save_json(self.incumbents, self.inc_file)
        # Info is not persisted now
        #info_strng = self._transform_dict_to_string(self.stat_information)
        #self._save_info_file(info_strng, info_file)

    def is_budget_exhausted(self):
        return time.time() - self.start_time > self.total_runtime

    def clean_files(self):
        self._clean_files([self.run_file, self.inc_file])


    #### INTERNAL METHODS ####

    def _add_incumbent(self, lst, config, time_point, information, config_origin):
        if 'time' in information.keys() or 'config' in information.keys() \
                or 'eval' in information.keys():
            raise ValueError("information should not contain the 'time' or 'config' key!")
        run = information.copy()
        run.update({
            'wallclock_time': time_point,
            'eval': len(lst) + 1,
            'incumbent': config,
            'config_origin': config_origin
        })
        lst.append(run)
        return run

    def _add_run(self, lst, config, time_point, information, config_origin):
        if 'time' in information.keys() or 'config' in information.keys()\
                or 'eval' in information.keys():
            raise ValueError("information should not contain the 'time' or 'config' key!")
        run = information.copy()
        run.update({
            'wallclock_time': time_point,
            'eval': len(lst) + 1,
            'config': config,
            'config_origin': config_origin
        })
        lst.append(run)
        return run

    def _read_incumbent_file(self, filename, config_space):
        incumbent_trajectory = []
        with open(filename) as fp:
            for line in fp:
                entry = json.loads(line)
                entry["config"] = self._convert_dict_to_config(
                    entry["config"], cs=config_space)
                incumbent_trajectory.append(entry)
        return incumbent_trajectory

    def _read_json_file(self, filename):
        trajectory = []
        with open(filename) as fp:
            for line in fp:
                entry = json.loads(line)
                trajectory.append(entry)
        return trajectory

    def _convert_dict_to_config(self, config_dict, config_space):
        # Method come from SMAC3

        config = Configuration(configuration_space=config_space, values=config_dict)
        config.origin = "External Trajectory"
        return config

    def _transform_dict_to_string(self, dct):
        strng = ""
        for key in dct:
            strng += (key + ": " + str(dct[key]) + "\n")
        return strng

    def _save_json(self, lst, destination_file):
        if not os.path.exists(destination_file):
            self._open_file(destination_file)

        with open(destination_file, "a") as fp:
            for row in lst:
                json.dump(row, fp, indent=4, sort_keys=True)
                fp.write("\n")

    def _save_info_file(self, strng, destination_file):
        if not os.path.exists(destination_file):
            self._open_file(destination_file)

        f = open(destination_file, 'w')
        f.write(strng)
        f.close()

    def _clean_files(self, files):
        for file in files:
            self._open_file(file)

    def _open_file(self, file):
        f = open(file, 'w')
        f.close()

    def _set_output_dir(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir


class WhiteboxStatistics(Statistics):

    def __init__(self, stamp, output_dir, information: dict, total_runtime=None, time_precision=None):
        super(WhiteboxStatistics, self).__init__(stamp, output_dir, information, total_runtime, time_precision)
        self.current_time = None

    def start_timer(self):
        super(WhiteboxStatistics, self).start_timer()
        self.current_time = time.time()

    def hack_time(self, time):
        self.current_time += time

    def get_time_point(self):
        if self.start_time == None or self.current_time == None:
            raise ValueError("Timer is not yet started!")
        return self.current_time - self.start_time

    def is_budget_exhausted(self):
        return self.current_time - self.start_time > self.total_runtime




