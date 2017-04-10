
from pc_smac.pc_smac.pipeline_space.pipeline_space import PipelineSpace
from pc_smac.pc_smac.pipeline_space.pipeline_step import PipelineStep, OneHotEncodingStep, ImputationStep, RescalingStep, \
    BalancingStep

from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.extra_rand_trees import ExtraTreesNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.fast_ica import FastICANode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.feature_agglomeration import FeatureAgglomerationNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.kitchen_sinks import RandomKitchenSinksNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.linear_svm import LinearSVMNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.nystroem_sampler import NystroemSamplerNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.polynomial import PolynomialFeaturesNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.random_trees_embedding import RandomTreesEmbeddingNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.select_percentile import SelectPercentileNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.select_rates import SelectRatesNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.no_preprocessing import NoPreprocessingNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.pca import PcaNode
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.kernel_pca import KernelPcaNode

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
from pc_smac.pc_smac.pipeline_space.classification_nodes.sgd import SGDNode
from pc_smac.pc_smac.pipeline_space.classification_nodes.random_forest import RandomForestNode


class PipelineSpaceBuilder(object):

    def __init__(self):

        self.preprocessor_nodes = {
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

        self.classifier_nodes = {
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

    def build_pipeline_space(self, preprocessor_names, classifier_names):
        prepr_nodes = [self.preprocessor_nodes[pname] for pname in preprocessor_names]
        class_nodes = [self.classifier_nodes[cname] for cname in classifier_names]

        pipeline_space = PipelineSpace()
        o_s = OneHotEncodingStep()
        i_s = ImputationStep()
        r_s = RescalingStep()
        b_s = BalancingStep()
        p_s = PipelineStep(name='feature_preprocessor', nodes=prepr_nodes, caching=True)
        c_s = PipelineStep(name='classifier', nodes=class_nodes)
        pipeline_space.add_pipeline_steps([o_s, i_s, r_s, b_s, p_s, c_s])

        return pipeline_space