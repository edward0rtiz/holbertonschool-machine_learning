#!/usr/bin/env python3
"""Script to calculate the sensitivity in a
    confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """
    Function to calculate the sensitivity
    Args:
        confusion: numpy.ndarray of shape
                    (classes, classes)
    Returns: numpy.ndarray of shape (classes,)
            containing the sensitivity of each class
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    TPR = TP / (TP + FN)
    return TPR
