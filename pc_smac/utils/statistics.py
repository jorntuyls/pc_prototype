
import os
import time
import json


class Statistics(object):

    def __init__(self, stamp, output_dir, information: dict, total_runtime, time_precision=10):
        self.stamp = stamp
        self.output_dir = output_dir
        self.stat_information = information
        self.total_runtime = total_runtime
        self.time_precision = time_precision

        self.runs = []
        self.incumbents = []

        self.start_time = None

    def start_timer(self):
        self.start_time = time.time()

    def get_time_point(self):
        if self.start_time == None:
            raise ValueError("Timer is not yet started!")
        return time.time() - self.start_time

    def add_run(self, config, information: dict):
        time_point = self.get_time_point()
        self._add_run(self.runs, config, time_point, information)

    def get_run_trajectory(self):
        return self.runs

    def add_new_incumbent(self, incumbent, information: dict):
        time_point = self.get_time_point()
        self._add_run(self.incumbents, incumbent, time_point, information)
        self._add_run(self.runs, incumbent, time_point, information)

    def get_incumbent_trajectory(self):
        return self.incumbents

    def save(self):
        # Create and clean files
        inc_file = self.output_dir + "statistics_incumbents_" + str(self.stamp) + ".json"
        inc_trans_file = self.output_dir + "statistics_incumbents_transformed_" + str(self.stamp) + ".json"
        run_file = self.output_dir + "statistics_runs_" + str(self.stamp) + ".json"
        run_trans_file = self.output_dir + "statistics_runs_transformed_" + str(self.stamp) + ".json"
        info_file = self.output_dir + "statistics_info_" + str(self.stamp) + ".json"
        self._clean_files([inc_file, inc_trans_file, run_file, run_trans_file, info_file])
        # Save info to files
        self._save_json(self.incumbents, inc_file)
        self._save_json(self.runs, run_file)
        info_strng = self._transform_dict_to_string(self.stat_information)
        self._save_info_file(info_strng, info_file)
        transformed_incumbents = self._transform_to_equidistant_time_points(self.incumbents)
        transformed_runs = self._transform_to_equidistant_time_points(self.runs)
        self._save_json(transformed_incumbents, inc_trans_file)
        self._save_json(transformed_runs, run_trans_file)


    #### INTERNAL METHODS ####

    def _add_run(self, lst, config, time_point, information):
        if 'time' in information.keys() or 'config' in information.keys()\
                or 'eval' in information.keys():
            raise ValueError("information should not contain the 'time' or 'config' key!")
        run = information.copy()
        run.update({
            'time': time_point,
            'eval': len(lst) + 1,
            'config': config,
        })
        lst.append(run)

    def _transform_to_equidistant_time_points(self, data_lst):
        new_data_lst = []
        time = 0.0
        while time <= self.total_runtime:
            time_count = 0
            for i in range(0, len(data_lst) - 1):
                elemi = data_lst[i]
                elemi1 = data_lst[i + 1]
                time_count += float(elemi['time'])
                if time < float(data_lst[0]['time']):
                    new_data_lst.append({
                        'time': time
                    })
                    break
                elif time > time_count and time <= time_count + float(elemi1['time']):
                    new_elem = elemi.copy()
                    new_elem['time'] = time
                    new_data_lst.append(new_elem)
                    break
                elif i == len(data_lst) - 2 and time > time_count + float(elemi1['time']):
                    new_elem = elemi1.copy()
                    new_elem['time'] = time
                    new_data_lst.append(new_elem)
                    break
            time += self.time_precision
        return new_data_lst

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
                json.dump(row, fp)
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




