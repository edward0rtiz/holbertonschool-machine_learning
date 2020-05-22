#!/usr/bin/env python3
"""
Train miniBatch
"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model using mini-batch gradient descent
    Args:
        X_train (np.ndarray): matrix (m, 784) containing the training data.
        Y_train (np.ndarray): matrix (m, 10) containing the training labels.
        X_valid (np.ndarray): matrix (m, 784) containing the validation data.
        Y_valid (np.ndarray): matrix (m, 10) containing the validation labels.
        batch_size (int): number of data points in a batch.
        epochs (int): number of times the training should pass through the
                      whole dataset.
        load_path (str): path from which to load the model.
        save_path (str): path to where the model should be saved after
                         training.
    Returns:
        str: the path where the model was saved.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]
        if m % batch_size == 0:
            complete = 1
            num_batches = int(m / batch_size)
        else:
            complete = 0
            num_batches = int(m / batch_size) + 1

        for i in range(epochs + 1):
            # Print the train previous values
            feed_t = {x: X_train, y: Y_train}
            feed_v = {x: X_valid, y: Y_valid}
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_t)
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_v)
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < epochs:
                X_shu, Y_shu = shuffle_data(X_train, Y_train)

                for k in range(num_batches):
                    if complete == 0 and k == num_batches - 1:
                        start = k * batch_size
                        X_minibatch = X_shu[start::]
                        Y_minibatch = Y_shu[start::]
                    else:
                        start = k * batch_size
                        end = (k * batch_size) + batch_size
                        X_minibatch = X_shu[start:end]
                        Y_minibatch = Y_shu[start:end]

                    feed_mb = {x: X_minibatch, y: Y_minibatch}
                    sess.run(train_op, feed_mb)

                    if (k + 1) % 100 == 0 and k != 0:
                        mb_c, mb_a = sess.run([loss, accuracy], feed_mb)
                        print("\tStep {}:".format(k + 1))
                        print("\t\tCost: {}".format(mb_c))
                        print("\t\tAccuracy: {}".format(mb_a))

        return saver.save(sess, save_path)