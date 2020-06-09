#!/usr/bin/env python3

import tensorflow as tf

def lenet5(x, y):


    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    #conv1
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                             padding='same', activation=activation,
                             kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)


    #conv2
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                             padding='valid', activation=activation,
                             kernel_initializer=init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)


    #flatten
    flatten = tf.layers.Flatten()(pool2)

    # FC 1
    FC1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    # FC 2
    FC2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(FC1)
    # FC 3
    FC3 = tf.layers.Dense(units=10, kernel_initializer=init)(FC2)

    # prediction
    y_pred = FC3

    # loss
    loss = tf.losses.softmax_cross_entropy(y, FC3)

    # train
    train = tf.train.AdamOptimizer().minimize(loss)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    y_pred = tf.nn.softmax(y_pred)

    return y_pred, train, loss, accuracy
