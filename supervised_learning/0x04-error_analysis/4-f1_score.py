#!/usr/bin/env python3
"""Script to calculate the f1-score in a
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


def f1_score(confusion):
    """
    Function to calculate the f1-score
    Args:
        confusion: numpy.ndarray of shape
                    (classes, classes)
    Returns: numpy.ndarray of shape (classes,)
            containing the f1-score of each class
    """
    p = precision(confusion)
    s = sensitivity(confusion)
    f1 = 2 * (p * s) / (p + s)
    return f1
