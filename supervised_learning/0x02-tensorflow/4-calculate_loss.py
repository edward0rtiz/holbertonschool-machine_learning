#!/usr/bin/env python3
"""Script for loss in tensorflow"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Method to calculate the cross-entropy loss
    of a prediction
    Args:
        y: input data type label in a placeholder
        y_pred: type tensor that contains the DNN prediction

    Returns:

    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
