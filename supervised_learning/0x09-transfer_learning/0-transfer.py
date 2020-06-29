#!/usr/bin/env python3
"""Transfer learning CIFAR-10 in densenet 121"""

import tensorflow as tf
import tensorflow.keras as K
import datetime
from tensorflow.keras.datasets import cifar10

# Global Variables
batch_size = 128
num_classes = 10
epochs = 32


def preprocess_data(X, Y):
    """
    Function that pre-processes the data for your model
    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
        where m is the number of data points
        Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes)
    return X_p, Y_p


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    inputs = K.Input(shape=(32, 32, 3))
    n_input = K.layers.Lambda(lambda image: tf.image.resize(image,
                                                            (155,
                                                             155)))(inputs)

    print('No Fine Tuning implemented')
    base_model = K.applications.densenet.DenseNet121(include_top=False,
                                                     weights='imagenet',
                                                     pooling='max',
                                                     input_tensor=n_input)

    FC = base_model.layers[-1].output
    FC = K.layers.Flatten()(FC)
    FC = K.layers.BatchNormalization()(FC)
    FC = K.layers.Dense(256, activation='relu')(FC)
    FC = K.layers.Dense(128, activation='relu')(FC)
    FC = K.layers.BatchNormalization()(FC)
    FC = K.layers.Dropout(0.2)(FC)
    FC = K.layers.Dense(10, activation='softmax')(FC)
    model = K.models.Model(inputs=inputs, outputs=FC)

    # model.summary()
    # REGULARIZATION AND OPTIMIZATION
    lrr = K.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                        factor=.01,
                                        patience=3,
                                        min_lr=1e-5)

    check_point = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                              monitor='val_loss',
                                              mode='min', save_best_only=True,
                                              verbose=1)

    early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               patience=10,
                                               verbose=1)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['acc'])

    # TENSOR BOARD
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=log_dir,
                                                   histogram_freq=1)

    print('No Data Augmentation implemented')
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[lrr,
                                   check_point,
                                   early_stopping,
                                   tensorboard_callback])

    # %tensorboard --logdir logs/fit
