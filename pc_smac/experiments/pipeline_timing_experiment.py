import argparse
import os

from smac.smbo.objective import average_cost

from pc_smac.pc_smac.config_space.config_space_builder import ConfigSpaceBuilder
from pc_smac.pc_smac.data_loader.data_loader import DataLoader
from pc_smac.pc_smac.pc_runhistory.pc_runhistory import PCRunHistory
from pc_smac.pc_smac.pipeline.pipeline_timer import PipelineTimer, CachedPipelineTimer
from pc_smac.pc_smac.pipeline_space.classification_nodes.adaboost import AdaBoostNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.bernoulli_nb import BernoulliNBNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.decision_tree import DecisionTreeNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.extra_trees import ExtraTreesClassifierNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.gaussian_nb import GaussianNBNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.gradient_boosting import GradientBoostingNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.k_nearest_neighbors import KNearestNeighborsNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.lda import LDANode
from pc_smac.pc_smac.pipeline_space.classification_nodes.liblinear_svc import LibLinear_SVC_Node
from pc_smac.pc_smac.pipeline_space.classification_nodes.libsvm_svc import LibSVM_SVC_Node
from pc_smac.pc_smac.pipeline_space.classification_nodes.multinomial_nb import MultinomialNBNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.passive_aggresive import PassiveAggresiveNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.qda import QDANode
from pc_smac.pc_smac.pipeline_space.classification_nodes.random_forest import RandomForestNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.sgd import SGDNode
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
from pc_smac.pc_smac.pipeline_space.pipeline_space import PipelineSpace
from pc_smac.pc_smac.pipeline_space.pipeline_step import PipelineStep, OneHotEncodingStep, ImputationStep, RescalingStep, \
    BalancingStep
from pc_smac.pc_smac.utils.statistics import Statistics

def run_experiment(data_id, location, output_dir, prepr_name=None, class_name=None, nb_configs=100, seed=None, cache_directory=None, downsampling=None):

    preprocessor_names = ['extra_rand_trees', 'fast_ica', 'feature_agglomeration', 'kernel_pca', 'kitchen_sinks', 'linear_svm',
                          'no_preprocessing', 'nystroem_sampler', 'pca', 'polynomial_features', 'rand_trees_embedding',
                          'select_percentile', 'select_rates']
    class_names = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                   'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggresive',
                   'qda', 'random_forest', 'sgd']
    #preprocessor_names = ['extra_rand_trees']

    preprocessor_nodes =  {
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

    classifier_nodes = {
        'adaboost': AdaBoostNode(),
        'bernoulli_nb': BernoulliNBNode(),
        'decision_tree': DecisionTreeNode(),
        'extra_trees': ExtraTreesClassifierNode(),
        'gaussian_nb': GaussianNBNode(),
        'gradient_boosting': GradientBoostingNode(),
        'k_nearest_neighbors': KNearestNeighborsNode(),
        'lda': LDANode(),
        'liblinear_svc': LibLinear_SVC_Node(),
        'libsvm_svc': LibSVM_SVC_Node(),
        'multinomial_nb': MultinomialNBNode(),
        'passive_aggresive': PassiveAggresiveNode(),
        'qda': QDANode(),
        'random_forest': RandomForestNode(),
        'sgd': SGDNode()
    }

    if prepr_name != None:
        prepr_nodes = [preprocessor_nodes[prepr_name]]
    else:
        prepr_nodes = []
        for prepr in preprocessor_names:
            prepr_nodes.append(preprocessor_nodes[prepr])
        prepr_name = 'all'

    if class_name != None:
        class_nodes = [classifier_nodes[class_name]]
    else:
        class_nodes = []
        for c_name in class_names:
            class_nodes.append(classifier_nodes[c_name])
        class_name = 'all'

    # ouput directory
    if output_dir == None:
        output_dir = os.path.dirname(os.path.abspath(__file__)) + "/results/"

    # Build pipeline space
    pipeline_space = PipelineSpace()
    o_s = OneHotEncodingStep()
    i_s = ImputationStep()
    r_s = RescalingStep()
    b_s = BalancingStep()
    p_s = PipelineStep(name='feature_preprocessor', nodes=prepr_nodes, caching=True)
    c_s = PipelineStep(name='classifier', nodes=class_nodes)
    pipeline_space.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s])

    # Build configuration space
    cs_builder = ConfigSpaceBuilder(pipeline_space)
    #print("SEED: {}".format(seed)) if seed else print("NOT SEED: {}".format(seed))
    config_space = cs_builder.build_config_space(seed=seed)

    # Sample configurations from configuration space
    rand_configs = config_space.sample_configuration(size=nb_configs) if nb_configs > 1 else [config_space.sample_configuration(size=nb_configs)]

    # Run the random configurations = pipelines on data set
    data_path = location + str(data_id) if location[-1] == "/" else location + "/" + str(data_id)
    data_set = data_path.split("/")[-1]
    output_dir = output_dir + data_set + "/" + str(prepr_name) + "_" + str(class_name) + "/" if output_dir[-1] == "/" \
                                            else output_dir + "/" + data_set + "/" + str(prepr_name) + "_" + str(class_name) + "/"
    stamp = data_set + "_seed_" + str(seed)
    run_experiment_on_data(stamp=stamp,
                           data_path=data_path,
                           output_dir=output_dir,
                           pipeline_space=pipeline_space,
                           configs=rand_configs,
                           cache_directory=cache_directory,
                           downsampling=downsampling)

def run_experiment_on_data(stamp, data_path, output_dir, pipeline_space, configs, cache_directory, downsampling):
    # Make own output directory for each data set

    print(output_dir)
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)
    except FileExistsError:
        pass

    # load data
    data_loader = DataLoader(data_path)
    data = data_loader.get_data()

    for i in range(0, len(configs)):
        ## Run with caching disabled ##
        new_stamp = stamp + "_nocaching_config_1"
        info = {
            'stamp': new_stamp,
            'caching': False,
            'downsampling': downsampling
        }
        statistics = Statistics(new_stamp, output_dir,
                                     information=info)
        # Build runhistory
        runhistory = PCRunHistory(average_cost)

        # Build pipeline runner
        pipeline_runner = PipelineTimer(data, pipeline_space, runhistory, statistics,
                       downsampling=downsampling)
        # Start timer
        statistics.start_timer()

        # Run pipeline
        pipeline_runner.run(config=configs[i], instance=1, seed=None)

        # Save statistics
        #statistics.save()

        ## Run with caching disabled second time ##
        new_stamp = stamp + "_nocaching_config_2"
        info = {
            'stamp': new_stamp,
            'caching': False,
            'downsampling': downsampling
        }
        statistics = Statistics(new_stamp, output_dir,
                                information=info)
        # Build runhistory
        runhistory = PCRunHistory(average_cost)

        # Build pipeline runner
        pipeline_runner = PipelineTimer(data, pipeline_space, runhistory, statistics,
                                        downsampling=downsampling)
        # Start timer
        statistics.start_timer()

        # Run pipeline
        pipeline_runner.run(config=configs[i], instance=1, seed=None)

        # Save statistics
        # statistics.save()


        ## Run with caching enabled first time ##
        new_stamp = stamp + "_caching_config_run_1"
        info = {
            'stamp': new_stamp,
            'caching': True,
            'cache_directory': cache_directory,
            'downsampling': downsampling
        }
        statistics = Statistics(new_stamp, output_dir,
                                information=info)
        # Build runhistory
        runhistory = PCRunHistory(average_cost)

        # Build pipeline runner
        cached_pipeline_runner = CachedPipelineTimer(data, pipeline_space, runhistory, statistics,
                                               cache_directory=cache_directory,
                                               downsampling=downsampling)
        # Start timer
        statistics.start_timer()

        # Run pipeline
        cached_pipeline_runner.run(config=configs[i], instance=1, seed=None)

        # Save statistics
        #statistics.save()


        ## Run with caching enabled second time ##
        new_stamp = stamp + "_caching_config_run_2"
        info = {
            'stamp': new_stamp,
            'caching': True,
            'cache_directory': cache_directory,
            'downsampling': downsampling
        }
        statistics = Statistics(new_stamp, output_dir,
                                information=info)

        # Give pipelinerunner new statistics object
        cached_pipeline_runner.update_statistics(statistics)

        # Start timer
        statistics.start_timer()

        # Run pipeline
        cached_pipeline_runner.run(config=configs[i], instance=1, seed=None)

        # clean cache
        cached_pipeline_runner.clean_cache()

        # Save statistics
        #statistics.save()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nbconfigs", type=int, help="Number of configurations")
    parser.add_argument("-d", "--dataid", type=str, help="Dataset id")
    parser.add_argument("-l", "--location", type=str, help="Dataset directory")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed for sampling configurations")
    parser.add_argument("-p", "--pname", type=str, default=None, help="Preprocessor name")
    parser.add_argument("-c", "--cname", type=str, default=None, help="Classifier name")
    parser.add_argument("-o", "--outputdir", type=str, default=None, help="Output directory")
    parser.add_argument("-ds", "--downsampling", type=int, default=None, help="Number of data points to downsample to")
    parser.add_argument("-cd", "--cachedir", type=str, default=None, help="Cache directory")
    args = parser.parse_args()

    return args




if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(data_id=args.dataid,
                   location=args.location,
                   output_dir=args.outputdir,
                   prepr_name=args.pname,
                   class_name=args.cname,
                   nb_configs=args.nbconfigs,
                   seed=args.seed,
                   cache_directory=args.cachedir,
                   downsampling=args.downsampling)
    # run_experiment(data_id="554_bac",
    #                location="/Users/jorntuyls/Documents/workspaces/thesis/data",
    #                output_dir="/Users/jorntuyls/Documents/workspaces/thesis/pc_smac/output",
    #                prepr_name="extra_rand_trees",
    #                class_name="sgd",
    #                nb_configs=1,
    #                seed=1,
    #                cache_directory="/Users/jorntuyls/Documents/workspaces/thesis/pc_smac/cache",
    #                downsampling=None)