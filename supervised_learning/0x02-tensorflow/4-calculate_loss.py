#!/usr/bin/env python3
"""Script for loss in tensorflow"""

import tensorflow as tf


def calculate_loss(y, y_pred):

    loss_f = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss_f

