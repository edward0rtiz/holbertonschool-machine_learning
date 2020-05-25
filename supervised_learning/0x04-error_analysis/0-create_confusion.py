#!/usr/bin/env python3
"""Script to create a confusion matrix"""

import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Function to create a confusion matrix
    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
                   m: is the number of data points
                   classes: is the number of classes
        logits: one-hot numpy.ndarray of shape (m, classes)
                containing the predicted labels

    Returns: a confusion numpy.ndarray of shape
            (classes, classes) with row indices representing
            the correct labels and column indices representing
            the predicted labels
    """
    return np.matmul(labels.T, logits)