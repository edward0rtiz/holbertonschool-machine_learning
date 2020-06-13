#!/usr/bin/env python3
"""Script for implement a Lenet5 using tensorflow"""

import tensorflow as tf


def lenet5(x, y):
    """
    Function to use a lenet5 with tensorflow
    Args:
        x: tf.placeholder of shape (m, 28, 28, 1)
           containing the input images for the network
           m: the number of images
        y: tf.placeholder of shape (m, 10) containing
           the one-hot labels for the network

    Returns: a tensor for the softmax activated output,
             training operation that utilizes Adam
             optimization (with default hyperparameters),
             tensor for the loss of the network,
             tensor for the accuracy of the network
    """
    # initialize global parameters
    init = tf.contrib.layers.variance_scaling_initializer()

    # Set the variable of activation 'relu'
    activation = tf.nn.relu

    # First CONVNET
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                             padding='same', activation=activation,
                             kernel_initializer=init)(x)
    # Pool net of CONV1
    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)

    # Second CONVNET
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                             padding='valid', activation=activation,
                             kernel_initializer=init)(pool1)
    # Pool net of CONV2
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)

    # Flatten the convolutional layers
    flatten = tf.layers.Flatten()(pool2)

    # Fully connected layer 1
    FC1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    # Fully connected layer 2
    FC2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(FC1)
    # Fully connected layer 3
    FC3 = tf.layers.Dense(units=10, kernel_initializer=init)(FC2)

    # Prediction variable
    y_pred = FC3

    y_pred = tf.nn.softmax(y_pred)

    # Loss function
    loss = tf.losses.softmax_cross_entropy(y, FC3)

    # Train function
    train = tf.train.AdamOptimizer().minimize(loss)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train, loss, accuracy
