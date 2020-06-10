#!/usr/bin/env python3
"""Script for Lenet5 using Keras"""

import tensorflow.keras as K


def lenet5(X):
    """
    Function to implement Lenet-5 using keras
    Args:
        X: X is a K.Input of shape (m, 28, 28, 1)
           containing the input images for the
    Returns: K.Model compiled to use Adam optimization
             (with default hyperparameters) and accuracy
             metrics
    """

    # initialize global parameters
    init = K.initializers.he_normal()

    # Set the variable of activation 'relu'
    activation = 'relu'

    # First CONVNET
    conv1 = K.layers.Conv2D(filters=6, kernel_size=5,
                            padding='same', activation=activation,
                            kernel_initializer=init)(X)
    # Pool net of CONV1
    pool1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)

    # Second CONVNET
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5,
                            padding='valid', activation=activation,
                            kernel_initializer=init)(pool1)
    # Pool net of CONV2
    pool2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)

    # Flatten the convolutional layers
    flatten = K.layers.Flatten()(pool2)

    # Fully connected layer 1
    FC1 = K.layers.Dense(units=120, activation=activation,
                         kernel_initializer=init)(flatten)
    # Fully connected layer 2
    FC2 = K.layers.Dense(units=84, activation=activation,
                         kernel_initializer=init)(FC1)
    # Fully connected layer 3
    FC3 = K.layers.Dense(units=10, kernel_initializer=init,
                         activation='softmax')(FC2)

    # Create Model
    model = K.models.Model(X, FC3)

    # Set Adam optimizer
    adam = K.optimizers.Adam()

    # Compile model
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
