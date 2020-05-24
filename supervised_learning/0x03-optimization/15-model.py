#!/usr/bin/model
"""Script to train a model with a multiple-object
    optimization theory
"""

import numpy as np
import tensorflow as tf


def shuffle_data(X, Y):
    """
    Function to shuffle data in a matrix
    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle
        Y: numpy.ndarray of shape (m, ny) to shuffle
    Returns: the shuffled X and Y matrices

    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    X_shuffled = X[shuffle]
    Y_shuffled = Y[shuffle]
    return X_shuffled, Y_shuffled


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


def calculate_accuracy(y, y_pred):
    """
    method to calculate the accuracy of a prediction in a DNN
    Args:
        y: input data type label in a placeholder
        y_pred: type tensor that contains the DNN prediction

    Returns: Prediction accuracy

    """
    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

def create_batch_norm_layer(prev, n, activation):
    """
    Function that normalized a batch in a DNN with Tf
    Args:
        prev: the activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function that should be used
                    on the output of the layer

    Returns: tensor of the activated output for the layer

    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None, kernel_initializer=init)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    offset = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    variance_epsilon = 1e-8

    normalization = tf.nn.batch_normalization(
        x_prev,
        mean,
        variance,
        offset,
        scale,
        variance_epsilon,
    )
    return activation(normalization)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Forward propagation method using TF
    Args:
        x: Input data (placeholder)
        layer_sizes: type list are the n nodes inside the layers
        activations: type list with the activation function per layer

    Returns: Prediction of a DNN

    """
    layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layer = create_batch_norm_layer(layer, layer_sizes[i], activations[i])
    return layer


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Function to train a DNN with TF RMSProp optimization
    Args:
        loss: loss of the network
        alpha: learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: small number to avoid division by zero

    Returns: Adam optimization operation

    """
    optimizer = tf.train.AdamOptimizer(alpha,
                                       beta1,
                                       beta2,
                                       epsilon).minimize(loss)
    return optimizer


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    learning rate decay operation in tensorflow using inverse time decay:
    Args:
        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes of gradient descent that should occur
                    before alpha is decayed further

    Returns:  learning rate decay operation

    """
    LRD = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                      decay_rate, staircase=True)
    return LRD


