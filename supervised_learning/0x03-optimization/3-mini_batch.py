#!/usr/bin/env python3
"""Script to train a model using mini batch"""


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.cpkt"):
    """
    Function to train a DNN using mini_batch gradient descent algorithm
    Args:
        X_train: numpy.ndarray of shape (m, 784)
                 contains training data
        Y_train: one-hot numpy.ndarray of shape (m, 10)
                 contains training labels
        X_valid: numpy.ndarray of shape (m, 784)
                 contains validation data
        Y_valid: one-hot numpy.ndarray of shape (m, 10)
                 contains validation labels
        batch_size: type int number of data points in a batch
        epochs: type int number of times the training should pass
                through the whole dataset
        load_path: path from which to load the model
        save_path: path to where the model should be saved after training

    Returns: path where the model was saved

    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]
        if (m % batch_size) == 0:
            num_minibatches = int(m / batch_size)
        else:
            num_minibatches = int(m / batch_size) + 1

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        for i_epoch in range(epochs):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy,
                                      feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i_epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i_epoch < epochs:
                Xs, Ys = shuffle_data(X_train, Y_train)
                for i in range(num_minibatches):
                    x_minbatch = Xs[i * batch_size: (i + 1) * batch_size]
                    y_minbatch = Ys[i * batch_size: (i + 1) * batch_size]
                    if i == num_minibatches - 1:
                        x_minbatch = Xs[i * batch_size:]
                        y_minbatch = Ys[i * batch_size:]

                    cost = sess.run(loss,
                                    feed_dict={x: x_minbatch, y: y_minbatch})
                    acc = sess.run(accuracy,
                                   feed_dict={x: x_minbatch, y: y_minbatch})

                    if (i % 100 == 0) and (i is not 0):
                        print("\tStep {}:".format(i))
                        print("\t\tCost {}:".format(cost))
                        print("\t\tAccuracy {}:".format(acc))
                    sess.run(train_op,
                             feed_dict={x: x_minbatch, y: y_minbatch})
        save_path = saver.save(sess, save_path)
    return save_path
