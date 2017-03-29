
import numpy as np
import scipy as sp
from pc_smac.pc_smac.utils.constants import MULTILABEL_CLASSIFICATION, \
    MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION
from pc_smac.pc_smac.utils.metrics_util import create_multiclass_solution, create_multiclass_prediction, binarize_predictions


def calculate_bac_score(solution, prediction, num_labels, task=BINARY_CLASSIFICATION):
    if task == MULTICLASS_CLASSIFICATION:
        prediction = create_multiclass_prediction(prediction, num_labels)

    return bac_metric(solution, prediction, task)

def bac_metric(solution, prediction, task=BINARY_CLASSIFICATION):
    """
    Compute the normalized balanced accuracy.

    The binarization and
    the normalization differ for the multi-label and multi-class case.
    :param solution:
    :param prediction:
    :param task:
    :return:
    """
    if task == BINARY_CLASSIFICATION:
        if len(solution.shape) == 1:
            # Solution won't be touched - no copy
            solution = solution.reshape((-1, 1))
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
        else:
            raise ValueError('Solution.shape %s' % solution.shape)

        if len(prediction.shape) == 1:
            # Solution won't be touched - no copy
            prediction = prediction.reshape((-1, 1))
        elif len(prediction.shape) == 2:
            if prediction.shape[1] > 2:
                raise ValueError('A prediction array with probability values '
                                 'for %d classes is not a binary '
                                 'classification problem' % prediction.shape[1])
            # Prediction will be copied into a new binary array - no copy
            prediction = prediction[:, 1].reshape((-1, 1))
        else:
            raise ValueError('Invalid prediction shape %s' % prediction.shape)

    elif task == MULTICLASS_CLASSIFICATION:
        if len(solution.shape) == 1:
            solution = create_multiclass_solution(solution, prediction)
        elif len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError('Solution array must only contain one class '
                                 'label, but contains %d' % solution.shape[1])
            else:
                solution = create_multiclass_solution(solution.reshape((-1, 1)),
                                                      prediction)
        else:
            raise ValueError('Solution.shape %s' % solution.shape)
    elif task == MULTILABEL_CLASSIFICATION:
        pass
    else:
        raise NotImplementedError('bac_metric does not support task type %s'
                                  % task)
    bin_prediction = binarize_predictions(prediction, task)


    fn = np.sum(np.multiply(solution, (1 - bin_prediction)), axis=0,
                dtype=float)
    tp = np.sum(np.multiply(solution, bin_prediction), axis=0, dtype=float)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = sp.maximum(eps, tp)
    pos_num = sp.maximum(eps, tp + fn)
    tpr = tp / pos_num  # true positive rate (sensitivity)

    if task in (BINARY_CLASSIFICATION, MULTILABEL_CLASSIFICATION):
        tn = np.sum(np.multiply((1 - solution), (1 - bin_prediction)),
                    axis=0, dtype=float)
        fp = np.sum(np.multiply((1 - solution), bin_prediction), axis=0,
                    dtype=float)
        tn = sp.maximum(eps, tn)
        neg_num = sp.maximum(eps, tn + fp)
        tnr = tn / neg_num  # true negative rate (specificity)
        bac = 0.5 * (tpr + tnr)
        base_bac = 0.5  # random predictions for binary case
    elif task == MULTICLASS_CLASSIFICATION:
        label_num = solution.shape[1]
        bac = tpr
        base_bac = 1. / label_num  # random predictions for multiclass case

    bac = np.mean(bac)  # average over all classes
    # Normalize: 0 for random, 1 for perfect
    # TODO Remove line before creating table
    #score = (bac - base_bac) / sp.maximum(eps, (1 - base_bac))
    return bac