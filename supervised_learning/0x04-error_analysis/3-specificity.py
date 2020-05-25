#!/usr/bin/env python3
"""Script to calculate the specificity in a
    confusion matrix
"""

import numpy as np


def specificity(confusion):
    """
    Function to calculate the specificity
    Args:
        confusion: numpy.ndarray of shape
                    (classes, classes)
    Returns: numpy.ndarray of shape (classes,)
            containing the specificity of each class
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - (FP + FN + TP)

    TNR = TN / (TN + FP)
    return TNR
