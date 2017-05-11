
import os
import argparse
import time
import json
import numpy as np
import tempfile

from sklearn.externals import joblib
from sklearn.externals.joblib import Memory

from pc_smac.pc_smac.data_loader.data_loader import DataLoader
from pc_smac.pc_smac.pipeline.cached_pipeline import CachedPipeline

from pc_smac.pc_smac.pipeline_space.data_preprocessing_nodes.one_hot_encoding import OneHotEncodeingNode

from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.extra_rand_trees import ExtraTreesNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.fast_ica import FastICANode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.feature_agglomeration import FeatureAgglomerationNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.kernel_pca import KernelPcaNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.kitchen_sinks import RandomKitchenSinksNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.linear_svm import LinearSVMNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.no_preprocessing import NoPreprocessingNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.nystroem_sampler import NystroemSamplerNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.pca import PcaNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.polynomial import PolynomialFeaturesNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.random_trees_embedding import RandomTreesEmbeddingNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.select_percentile import SelectPercentileNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.select_rates import SelectRatesNode

from pc_smac.pc_smac.pipeline_space.classification_nodes.sgd import SGDNode


def run_caching_experiment(stamp, data_path, data_set_id, prepr_name, cache_dir=None, output_dir=None):
    data_path = os.path.join(data_path, data_set_id)

    if cache_dir == None:
        cache_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir == None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    cache_dir = os.path.join(cache_dir, data_set_id + "/" + prepr_name)
    output_dir = os.path.join(output_dir, data_set_id + "/" + prepr_name)

    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    except FileExistsError:
        pass

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except FileExistsError:
        pass

    preprocessor_nodes = {
        'extra_rand_trees': ExtraTreesNode(),
        'fast_ica': FastICANode(),
        'feature_agglomeration': FeatureAgglomerationNode(),
        'kernel_pca': KernelPcaNode(),
        'kitchen_sinks': RandomKitchenSinksNode(),
        'linear_svm': LinearSVMNode(),
        'no_preprocessing': NoPreprocessingNode(),
        'nystroem_sampler': NystroemSamplerNode(),
        'pca': PcaNode(),
        'polynomial_features': PolynomialFeaturesNode(),
        'rand_trees_embedding': RandomTreesEmbeddingNode(),
        'select_percentile': SelectPercentileNode(),
        'select_rates': SelectRatesNode()
    }

    data_loader = DataLoader(data_path)
    data = data_loader.get_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    print(X_train.shape, y_train.shape)


    output = dict()


    fp_node = preprocessor_nodes[prepr_name]
    transformer_name, transformer = fp_node.initialize_algorithm(hyperparameters={})
    print(transformer)
    print("Start")
    start_time = time.time()
    transformer_after = transformer.fit(X_train, y_train)
    timing = time.time() - start_time
    print("Timing transform: {}".format(timing))

    fp_node = preprocessor_nodes[prepr_name]
    transformer_name, transformer = fp_node.initialize_algorithm(hyperparameters={})
    print(transformer)
    print("Start")
    start_time = time.time()
    transformer_after = transformer.fit(X_train, y_train)
    timing = time.time() - start_time
    print("Timing transform: {}".format(timing))
    output['fit_timing_alone'] = timing

    #### CACHE transformer ####
    fp_node = preprocessor_nodes[prepr_name]
    transformer_name_2, transformer_2 = fp_node.initialize_algorithm(hyperparameters={})
    print("Start")
    print(transformer_name_2, transformer_2)
    start_time = time.time()
    transformer_after = transformer_2.fit(X_train, y_train)
    joblib.dump(transformer_after, os.path.join(cache_dir, stamp + '_transform.pkl'))
    timing = time.time() - start_time
    print("Timing transform and persist: {}".format(timing))
    output['cache_transformer_fit_timing_with_cache'] = timing


    print("Start")
    start_time = time.time()
    transformer_after = joblib.load(os.path.join(cache_dir, stamp + '_transform.pkl'))
    load_timing = time.time() - start_time
    X1 = transformer_after.transform(X_train)
    timing = time.time() - start_time
    print("Timing load transformer and transform : {}".format(timing))
    output['cache_transformer_load_cache'] = load_timing
    output['cache_transformer_load_cache_and_transform'] = timing

    #### CACHE result ####

    fp_node = preprocessor_nodes[prepr_name]
    transformer_name_3, transformer_3 = fp_node.initialize_algorithm(hyperparameters={})
    print("Start")
    print(transformer_name_3, transformer_3)
    start_time = time.time()
    X = transformer_3.fit(X_train, y_train).transform(X_train)
    joblib.dump(X, os.path.join(cache_dir, stamp + '_X.pkl'))
    timing = time.time() - start_time
    print("Timing transform and persist result: {}".format(timing))
    output['cache_result_fit_timing_with_cache'] = timing

    print("Start")
    start_time = time.time()
    X2 = joblib.load(os.path.join(cache_dir, stamp + '_X.pkl'))
    timing = time.time() - start_time
    print("Timing load result and transform : {}".format(timing))
    output['cache_result_load_cache'] = timing

    save_json([output], os.path.join(output_dir, 'result_' + stamp + '.json'))

    print(X1 == X2)

def run_float32_experiment():
    cachedir = tempfile.mkdtemp(dir="/Users/jorntuyls/Desktop/cache", prefix="cache_")

    # CACHED RUN
    one_hot_encoder = OneHotEncodeingNode().initialize_algorithm(hyperparameters={})
    kitchen_sinks = RandomKitchenSinksNode().initialize_algorithm(hyperparameters={})
    sgd = SGDNode().initialize_algorithm(hyperparameters={})
    steps = [one_hot_encoder, kitchen_sinks, sgd]

    pipeline = CachedPipeline(steps=steps, cached_step_names=["feature_preprocessor:kitchen_sinks"], memory=Memory(cachedir=cachedir, verbose=0))

    X = np.random.randn(50000, 5000)

    print("fit")
    start_time = time.time()
    pipeline.fit(X)
    print("Fit timing: {}".format(time.time() - start_time))
    print("after fit")

    # CACHE
    one_hot_encoder = OneHotEncodeingNode().initialize_algorithm(hyperparameters={})
    kitchen_sinks = RandomKitchenSinksNode().initialize_algorithm(hyperparameters={})
    sgd = SGDNode().initialize_algorithm(hyperparameters={})
    steps = [one_hot_encoder, kitchen_sinks, sgd]

    pipeline = CachedPipeline(steps=steps, cached_step_names=["feature_preprocessor:kitchen_sinks"],
                              memory=Memory(cachedir=cachedir, verbose=0))

    print("fit cache")
    start_time = time.time()
    pipeline.fit(X)
    print("Load result timing: {}".format(time.time() - start_time))
    print("after fit cache")

    # NO CACHE HIT
    one_hot_encoder = OneHotEncodeingNode().initialize_algorithm(hyperparameters={"use_minimum_fraction": "False"})
    kitchen_sinks = RandomKitchenSinksNode().initialize_algorithm(hyperparameters={})
    sgd = SGDNode().initialize_algorithm(hyperparameters={})
    steps = [one_hot_encoder, kitchen_sinks, sgd]

    pipeline = CachedPipeline(steps=steps, cached_step_names=["feature_preprocessor:kitchen_sinks"],
                              memory=Memory(cachedir=cachedir, verbose=0))
    print("fit other encoder")
    start_time = time.time()
    pipeline.fit(X)
    print("Fit other encoder timing: {}".format(time.time() - start_time))

    # NO CACHE HIT 2
    one_hot_encoder = OneHotEncodeingNode().initialize_algorithm(hyperparameters={})
    kitchen_sinks = RandomKitchenSinksNode().initialize_algorithm(hyperparameters={})
    sgd = SGDNode().initialize_algorithm(hyperparameters={})
    steps = [one_hot_encoder, kitchen_sinks, sgd]

    pipeline = CachedPipeline(steps=steps, cached_step_names=["feature_preprocessor:kitchen_sinks"],
                              memory=Memory(cachedir=cachedir, verbose=0))

    X = np.random.randn(20000, 2000)

    print("fit other encoder 2")
    start_time = time.time()
    pipeline.fit(X)
    print("Fit other encoder timing 2: {}".format(time.time() - start_time))


def save_json(lst, destination_file):
    if not os.path.exists(destination_file):
        open_file(destination_file)

    with open(destination_file, "a") as fp:
        for row in lst:
            json.dump(row, fp, indent=4, sort_keys=True)
            fp.write("\n")

def open_file(file):
    f = open(file, 'w')
    f.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--location", type=str, help="Dataset directory")
    parser.add_argument("-d", "--dataid", type=str, help="Dataset directory")
    parser.add_argument("-s", "--stamp", type=str, default="", help="Stamp")
    parser.add_argument("-p", "--pname", type=str, default=None, help="Preprocessor name")
    parser.add_argument("-o", "--outputdir", type=str, default=None, help="Output directory")
    parser.add_argument("-ds", "--downsampling", type=int, default=None,
                        help="Number of data points to downsample to")
    parser.add_argument("-cd", "--cachedir", type=str, default=None, help="Cache directory")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # args = parse_arguments()
    # run_caching_experiment(stamp=args.stamp,
    #                        data_path=args.location,
    #                        data_set_id=args.dataid,
    #                        prepr_name=args.pname,
    #                        cache_dir=args.cachedir,
    #                        output_dir=args.outputdir)
    run_float32_experiment()



