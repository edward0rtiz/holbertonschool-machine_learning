#!/usr/bin/env python3
"""Script to calculate the f1-score in a
    confusion matrix
"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


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
