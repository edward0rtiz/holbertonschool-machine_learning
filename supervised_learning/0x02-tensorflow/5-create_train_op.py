#!/usr/bin/env python3
"""Script to train a model in tensorflow
   using gradient descent optimizer
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    method to train a model in a DNN with tf.
    Args:
        loss: cross-entropy loss function
        alpha: learning rate

    Returns: Trained operation of de DNN using gradient descent

    """
    opt = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return opt
