#!/usr/bin/env python3
"""Script to train a model in tensorflow
   using gradient descent optimizer
"""

import tensorflow as tf


def create_train_op(loss, alpha):

    opt = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return opt
