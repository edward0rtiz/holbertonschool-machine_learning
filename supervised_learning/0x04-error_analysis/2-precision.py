#!/usr/bin/env python3
"""Script to calculate the precision in a
    confusion matrix
"""

import numpy as np


def precision(confusion):
    """
    Function to calculate the precision
    Args:
        confusion: numpy.ndarray of shape
                    (classes, classes)
    Returns: numpy.ndarray of shape (classes,)
            containing the precision of each class
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    PPV = TP / (TP + FP)
    return PPV
