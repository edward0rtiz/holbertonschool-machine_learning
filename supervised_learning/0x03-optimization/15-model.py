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
    x = tf.layers.Dense(units=n, kernel_initializer=init)
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
    if activation is None:
        return normalization
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


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """

    Args:
        Data_train: tuple containing the training inputs and
                    training labels, respectively
        Data_valid:  tuple containing the validation inputs and
                    validation labels, respectively
        layers:  list containing the number of nodes in each layer of
                the network
        activations: list containing the activation functions used
                    for each layer of the network
        alpha: learning rate
        beta1: weight for the first moment of Adam Optimization
        beta2: weight for the second moment of Adam Optimization
        epsilon: small number used to avoid division by zero
        decay_rate: decay rate for inverse time decay of the learning rate
        batch_size: number of data points that should be in a mini-batch
        epochs: number of times the training should pass through the whole
                dataset
        save_path: path where the model should be saved to

    Returns:  path where the model was saved
    """
    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x = tf.placeholder(tf.float32, shape=(None,
                                          Data_train[0].shape[1]), name='x')
    y = tf.placeholder(tf.float32, shape=(None,
                                          Data_train[1].shape[1]), name='y')

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        if (m % batch_size) == 0:
            num_minibatches = int(m / batch_size)
            check = 1
        else:
            num_minibatches = int(m / batch_size) + 1
            check = 0

        for epoch in range(epochs + 1):
            feed_train = {x: X_train, y: Y_train}
            feed_valid = {x: X_valid, y: Y_valid}
            train_cost = sess.run(loss, feed_dict=feed_train)
            train_accuracy = sess.run(accuracy, feed_dict=feed_train)
            valid_cost = sess.run(loss, feed_dict=feed_valid)
            valid_accuracy = sess.run(accuracy, feed_dict=feed_valid)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                Xs, Ys = shuffle_data(X_train, Y_train)

                for step_number in range(num_minibatches):
                    start = step_number * batch_size
                    end = (step_number + 1) * batch_size
                    if check == 0 and step_number == num_minibatches - 1:
                        x_minbatch = Xs[start::]
                        y_minbatch = Ys[start::]
                    else:
                        x_minbatch = Xs[start:end]
                        y_minbatch = Ys[start:end]

                    feed_mini = {x: x_minbatch, y: y_minbatch}
                    sess.run(train_op, feed_dict=feed_mini)

                    if ((step_number + 1) % 100 == 0) and (step_number != 0):
                        step_cost = sess.run(loss, feed_dict=feed_mini)
                        step_accuracy = sess.run(accuracy, feed_dict=feed_mini)
                        print("\tStep {}:".format(step_number + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))
            save_path = saver.save(sess, save_path)
    return save_path
