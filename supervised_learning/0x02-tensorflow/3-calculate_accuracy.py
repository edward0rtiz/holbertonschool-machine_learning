#!/usr/bin/env python3
"""Script for accuracy in tensor flow"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    method to calculate the accuracy of a prediction in a DNN
    Args:
        y: input data type label in a placeholder
        y_pred: type tenser that contains the DNN prediction

    Returns: Prediction accuracy

    """
    correct_prediction = tf.equal(tf.argmax(y, axis=1),
                                  tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
