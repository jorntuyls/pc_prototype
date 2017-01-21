
import csv
import os

def save_trajectory_to_plotting_format(trajectory, destination="results", filename="validationResults-traj-run-1.csv"):

        if not os.path.exists(destination):
            os.makedirs(destination)

        filename = destination + "/" + filename

        with open(filename, 'w') as csvfile:
            fieldnames = ['Time', 'Training Performance', 'Test Performance', 'AC Overhead Time',
                     'Validation Configuration ID']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for traj in trajectory:
                writer.writerow({'Time': traj['wallclock_time'],
                                 'Training Performance': traj['training_performance'],
                                 'Test Performance': traj['test_performance'],
                                 'AC Overhead Time': 0,
                                 'Validation Configuration ID': 1})

def append_dict_to_csv(dict, keys=None, destination="results", filename="dict_to_csv.csv"):
    if not os.path.exists(destination):
        os.makedirs(destination)

    filename = destination + "/" + filename
    if not os.path.exists(filename):
        with open(filename, 'w') as csvfile:
            fieldnames = keys if keys != None else dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #writer.writeheader()

    with open(filename, 'a') as csvfile:
        fieldnames = keys if keys != None else dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(dict)


# TODO [Temporary] Move this out of here
def transform(destination="results", filename="pipeline_runner_information_no_caching.csv"):
    filename = destination + "/" + filename

    d = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['runtime', 'cost', 'total_evaluations', 'cache_hits'])
        for row in reader:
            d.append(row)

    new_d = []
    time = 0.0
    total_time = 1080
    while time <= total_time:
        time_count = 0
        for i in range(0, len(d)-1):
            elemi = d[i]
            elemi1 = d[i+1]
            time_count += float(elemi['runtime'])
            if time < float(d[0]['runtime']):
                new_d.append({'time': 0.0,
                                'evals': 0,
                                'cost': 1,
                                'total_evaluations': 0,
                                'cache_hits': 0
                })
                break
            elif time > time_count and time <= time_count + float(elemi1['runtime']):
                new_d.append({
                    'time': time,
                    'evals': i,
                    'cost': elemi['cost'],
                    'total_evaluations': elemi['total_evaluations'],
                    'cache_hits': elemi['cache_hits']
                })
                break
            elif i == len(d)-2 and time > time_count + float(elemi1['runtime']):
                new_d.append({
                    'time': time,
                    'evals': i+1,
                    'cost': elemi1['cost'],
                    'total_evaluations': elemi1['total_evaluations'],
                    'cache_hits': elemi1['cache_hits']
                })
                break
        time += 10

    write_list_dicts(new_d, ['time', 'evals', 'cost', 'total_evaluations', 'cache_hits'])

def write_list_dicts(lst, keys, destination="results", filename="pipeline_runner_information_structured.csv" ):
    if not os.path.exists(destination):
        os.makedirs(destination)
    filename = destination + "/" + filename
    with open(filename, 'w') as csvfile:
        fieldnames = keys
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for row in lst:
            writer.writerow(row)


def save_trajectory_for_plotting(trajectory, wallclock_limit, plot_time, caching, run_counter=1):
    filename = "validationResults-traj-caching=" + str(caching) + "-run=" + str(run_counter) + ".csv"
    destination = os.path.dirname(os.path.abspath(__file__)) + "/results"
    # Create new trajectory file with choosen timestamps
    traj_0 = trajectory[0]
    new_trajectory = [{
        'wallclock_time': 0.0,
        'training_performance': traj_0['cost'],
        'test_performance': traj_0['test_performance']
    }]

    time = plot_time
    while time <= wallclock_limit:
        t = None
        # find trajectory element that is the incumbent at time 'time'
        for i in range(0, len(trajectory) - 1):
            traj_i = trajectory[i]
            traj_i1 = trajectory[i + 1]
            if time >= traj_i['wallclock_time'] and time < traj_i1['wallclock_time']:
                t = traj_i
                break
            elif time > trajectory[-1]['wallclock_time'] \
                    :
                t = trajectory[-1]
        if t:
            new_trajectory.append({
                'wallclock_time': time,
                'training_performance': t['cost'],
                'test_performance': t['test_performance']
            })
        else:
            raise ValueError("trajectory element should never be none")

        time += plot_time
    print(new_trajectory)
    save_trajectory_to_plotting_format(new_trajectory, destination=destination, filename=filename)
