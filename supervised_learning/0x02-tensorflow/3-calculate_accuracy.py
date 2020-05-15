#!/usr/bin/env python3
"""Script for accuracy in tensor flow"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):

    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
