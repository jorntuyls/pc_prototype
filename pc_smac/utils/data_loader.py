
# The code in this file is heavily based on autosklearn.data.CompetitionDataManager
#    you can find autosklearn here: https://github.com/automl/auto-sklearn


import os
import re

import numpy as np

from utils.constants import MULTILABEL_CLASSIFICATION, \
    STRING_TO_TASK_TYPES, MULTICLASS_CLASSIFICATION, STRING_TO_METRIC
from utils.data_utils import convert_to_num


class DataLoader:

    def __init__(self, name):

        input_dir, name = self._process_name(name)

        self.info_file = os.path.join(input_dir, name + '_public.info')
        self.data_file_train = os.path.join(input_dir, name + '_train.data')
        self.data_file_valid = os.path.join(input_dir, name + '_valid.data')
        self.data_file_test = os.path.join(input_dir, name + '_test.data')
        self.label_file_train = os.path.join(input_dir, name + '_train.solution')
        self.label_file_valid = os.path.join(input_dir, name + '_valid.solution')
        self.label_file_test = os.path.join(input_dir, name + '_test.solution')

        self.info = self._get_info(self.info_file, self.data_file_train)

        feat_type_file = os.path.join(input_dir, name + '_feat.type')
        self.feat_type = self._load_type(feat_type_file)

        self.data = None


    def get_data(self, encode_labels=False, max_memory_in_mb=1048576):
        if self.data == None:
            self.data = self._load_data(encode_labels, max_memory_in_mb)
        return self.data

    #### Internal methods ####

    def _load_data(self, encode_labels=False, max_memory_in_mb=1048576):
        # apply memory limit here for really large training sets
        Xtr = self._load_data_file(self.data_file_train,
            self.info['train_num'],
            max_memory_in_mb=max_memory_in_mb)
        Ytr = self._load_label_file(
            self.label_file_train,
            self.info['train_num'])
        # no restriction here
        Xva = self._load_data_file(
            self.data_file_valid,
            self.info['valid_num'],
            max_memory_in_mb=1048576)
        Yva = self._load_label_file(
            self.label_file_valid,
            self.info['valid_num'])
        Xte = self._load_data_file(
            self.data_file_test,
            self.info['test_num'],
            max_memory_in_mb=1048576)
        Yte = self._load_label_file(
            self.label_file_test,
            self.info['test_num'])

        # update the info in case the data has been cut off
        self.info['train_num'] = Xtr.shape[0]

        # TODO
        #if encode_labels:
        #    self.perform1HotEncoding()

        return {"X_train": Xtr, "y_train": Ytr,
                "X_valid": Xva, "y_valid": Yva,
                "X_test": Xte, "y_test": Yte}

    def _get_info(self, info_file, data_file):
        """Get all information {key = value} pairs from the filename
        (public.info file), if it exists, otherwise, output default values"""
        info = {}
        if os.path.exists(info_file):
            info.update(self._get_info_from_file(info_file))
            # print('Info file found : ' + os.path.abspath(filename))
            # Finds the data format ('dense', 'sparse', or 'sparse_binary')
            if 'format' not in info.keys():
                info.update(self._get_format_data(data_file, info))
        else:
            raise NotImplementedError('The user must always provide an info '
                                      'file.')

        info['task'] = STRING_TO_TASK_TYPES[info['task']]
        info['metric'] = STRING_TO_METRIC[info['metric']]

        return info

    def _load_data_file(self, filename, num_points, max_memory_in_mb):
        """Get the data from a text file in one of 3 formats:
        matrix, sparse, binary_sparse"""
        # if verbose:
        # print('========= Reading ' + filename)
        # start = time.time()
        if 'format' not in self.info:
            raise ValueError("There should be a format in self.info \
                                    before evoking this function")

        data_func = {
            'dense': data_dense,
            'sparse': data_sparse,
            'sparse_binary': data_binary_sparse
        }

        data = data_func[self.info['format']](filename, self.feat_type)

        # end = time.time()
        # if verbose:
        #     print('[+] Success in %5.2f sec' % (end - start))
        return data

    def _load_label_file(self, filename, num_points):
        """Get the solution/truth values."""
        # if verbose:
        # print('========= Reading ' + filename)
        # start = time.time()

        if self.info['task'] == MULTILABEL_CLASSIFICATION:
            label = load_labels(filename)
        elif self.info['task'] == MULTICLASS_CLASSIFICATION:
            label = convert_to_num(load_labels(filename))
        else:
            label = np.ravel(load_labels(filename))  # get a column vector

        # end = time.time()
        # if verbose:
        #     print('[+] Success in %5.2f sec' % (end - start))
        return label

    def _get_info_from_file(self, filename):
        """Get all information {key = value} pairs from public.info file"""
        info = {}
        with open(filename, 'r') as info_file:
            lines = info_file.readlines()
            features_list = list(
                map(lambda x: tuple(x.strip("\'").split(' = ')), lines))
            for (key, value) in features_list:
                info[key] = value.rstrip().strip("'").strip(' ')
                if info[key].isdigit(
                ):  # if we have a number, we want it to be an integer
                    info[key] = int(info[key])
        return info

    def _get_format_data(self, filename, info):
        """Get the data format directly from the data file (in case we do not
        have an info file)"""
        # Default
        info['format'] = 'dense'
        info['is_sparse'] = 0

        if 'is_sparse' in info.keys():
            if info['is_sparse'] == 0:
                info['format'] = 'dense'
            else:
                data = read_first_line(filename)
                if ':' in data[0]:
                    info['format'] = 'sparse'
                else:
                    info['format'] = 'sparse_binary'
        else:
            data = file_to_array(filename)
            if ':' in data[0][0]:
                info['is_sparse'] = 1
                info['format'] = 'sparse'
            else:
                nbr_columns = len(data[0])
                for row in range(len(data)):
                    if len(data[row]) != nbr_columns:
                        info['format'] = 'sparse_binary'
        return info

    def _load_type(self, filename):
        """Get the variable types."""
        if os.path.isfile(filename):
            type_list = file_to_array(filename)
        else:
            n = self.info['feat_num']
            type_list = [self.info['feat_type']] * n
        type_list = np.array(type_list).ravel()
        return type_list

    def _process_name(self, name):
        if name.endswith("/"):
            name = name[:-1]
        input_dir = os.path.dirname(name)
        if not input_dir:
            input_dir = "."
        name = os.path.basename(name)
        input_dir = os.path.join(input_dir, name)

        return input_dir, name



def data_dense(filename, feat_type=None):
    # The 2nd parameter makes possible a using of the 3 functions of data
    # reading (data, data_sparse, data_binary_sparse) without changing
    # parameters

    # This code is based on scipy.io.arff.arff_load
    r_comment = re.compile(r'^%')
    # Match an empty line
    r_empty = re.compile(r'^\s+$')
    descr = [(str(i), np.float32) for i in range(len(feat_type))]

    def generator(row_iter, delim=','):
        # Copied from scipy.io.arff.arffread
        raw = next(row_iter)
        while r_empty.match(raw) or r_comment.match(raw):
            raw = next(row_iter)

        # 'compiling' the range since it does not change
        # Note, I have already tried zipping the converters and
        # row elements and got slightly worse performance.
        elems = list(range(len(feat_type)))

        row = raw.split(delim)
        yield tuple([row[i] for i in elems])
        for raw in row_iter:
            while r_comment.match(raw) or r_empty.match(raw):
                raw = next(row_iter)
            row = raw.split(delim)
            # yield tuple([np.float64(row[i]) for i in elems])
            yield tuple([row[i] for i in elems])

    with open(filename) as fh:
        a = generator(fh, delim=' ')
        # No error should happen here: it is a bug otherwise
        data = np.fromiter(a, descr)

        data = data.view(np.float32).reshape((len(data), -1))
        return data


def data_sparse(filename, feat_type):
    # This function takes as argument a file representing a sparse matrix
    # sparse_matrix[i][j] = "a:b" means matrix[i][a] = b
    # It converts it into a numpy array, using sparse_list_to_array function,
    # and returns this array
    sparse_list = sparse_file_to_sparse_list(filename)
    return sparse_list_to_csr_sparse(sparse_list, len(feat_type))


def data_binary_sparse(filename, feat_type):
    # This function takes as an argument a file representing a binary sparse
    # matrix
    # binary_sparse_matrix[i][j] = a means matrix[i][j] = 1
    # It converts it into a numpy array an returns this array.

    inner_data = file_to_array(filename)
    nbr_samples = len(inner_data)
    # the construction is easier w/ dok_sparse
    dok_sparse = scipy.sparse.dok_matrix((nbr_samples, len(feat_type)))
    # print('Converting {} to dok sparse matrix'.format(filename))
    for row in range(nbr_samples):
        for feature in inner_data[row]:
            dok_sparse[row, int(feature) - 1] = 1
    # print('Converting {} to csr sparse matrix'.format(filename))
    return dok_sparse.tocsr()


def file_to_array(filename):
    # Converts a file to a list of list of STRING; It differs from
    # np.genfromtxt in that the number of columns doesn't need to be constant
    with open(filename, 'r') as data_file:
        # if verbose:
        # print('Reading {}...'.format(filename))
        lines = data_file.readlines()
        # if verbose:
        #     print('Converting {} to correct array...'.format(filename))
        data = [lines[i].strip().split() for i in range(len(lines))]
    return data


def read_first_line(filename):
    # Read fist line of file
    with open(filename, 'r') as data_file:
        line = data_file.readline()
        data = line.strip().split()
    return data


def sparse_file_to_sparse_list(filename):
    # Converts a sparse data file to a sparse list, so that:
    # sparse_list[i][j] = (a,b) means matrix[i][a]=b
    data_file = open(filename, 'r')
    # if verbose:
    # print('Reading {}...'.format(filename))
    lines = data_file.readlines()
    # if verbose:
    #     print('Converting {} to correct array')
    data = [lines[i].split(' ') for i in range(len(lines))]
    # if verbose:
    #     print('Converting {} to sparse list'.format(filename))

    _converter = lambda a_: (int(a_[0]), np.float32(float(a_[1])))
    return [[_converter(data[i][j].rstrip().split(':'))
             for j in range(len(data[i])) if data[i][j] != '\n']
            for i in range(len(data))]


def sparse_list_to_csr_sparse(sparse_list, nbr_features):
    # This function takes as argument a matrix of tuple representing a sparse
    # matrix and the number of features.
    # sparse_list[i][j] = (a,b) means matrix[i][a]=b
    # It converts it into a scipy csr sparse matrix
    nbr_samples = len(sparse_list)
    # construction easier w/ dok_sparse...
    dok_sparse = scipy.sparse.dok_matrix((nbr_samples, nbr_features),
                                         dtype=np.float32)
    # if verbose:
    # print('\tConverting sparse list to dok sparse matrix')
    for row in range(nbr_samples):
        for column in range(len(sparse_list[row])):
            (feature, value) = sparse_list[row][column]
            dok_sparse[row, feature - 1] = value
    # if verbose:
    #    print('\tConverting dok sparse matrix to csr sparse matrix')
    #     # but csr better for shuffling data or other tricks
    return dok_sparse.tocsr()


def load_labels(filename):
    return np.genfromtxt(filename, dtype=np.float64)
