#!/usr/bin/env python3
"""Script to train a model using keras"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """
    Function to train a model using keras and LRD
    Args:
        network: model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing
                the labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data: data to validate the model with, if not None
        early_stopping: boolean that indicates whether early stopping
                        should be used
        patience: patience used for early stopping
        learning_rate_decay: boolean that indicates whether learning rate decay
                             should be used
        alpha: initial learning rate
        decay_rate: decay rate
        verbose: boolean that determines if output should be printed during
                 training
        shuffle: boolean that determines whether to shuffle the batches every
                 epoch.
    Returns: History object generated after training the model

    """

    def scheduler(epoch):
        """
        Function o get the learning reate of each epoch
        Args:
            epoch: umber of passes through data for mini-batch gradient descent

        Returns:

        """
        return alpha / (1 + decay_rate * epoch)

    custom_callbacks = []
    ES = K.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                   patience=patience)
    LRD = K.callbacks.LearningRateScheduler(scheduler, verbose=1)

    if validation_data and early_stopping:
        custom_callbacks.append(ES)
    if validation_data and learning_rate_decay:
        custom_callbacks.append(LRD)
    if save_best:
        save = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        custom_callbacks.append(save)

    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          callbacks=custom_callbacks,
                          verbose=verbose, shuffle=shuffle)
    return history
