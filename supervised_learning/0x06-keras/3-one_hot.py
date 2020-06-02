#!/usr/bin/env python3
"""Script to laber vector to one hot in Keras"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Function to implement one-hot using keras
    Args:
        labels: labels of the set
        classes: classes of the set

    Returns: the one-hot matrix

    """
    One_hot = K.utils.to_categorical(labels, num_classes=classes)
    return One_hot
