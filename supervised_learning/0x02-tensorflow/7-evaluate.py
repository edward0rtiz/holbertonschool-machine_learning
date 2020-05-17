#!/usr/bin/env python3
"""Script that evaluates a DNN in tf"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Method to evaluate a DNN
    Args:
        X: input data
        Y: labels from X
        save_path: Location of trained model

    Returns: Evaluated model

    """
    with tf.Session as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(sess, save_path)
        x = tf.get_collection("x")
        y = tf.get_collection("y")
        y_pred = tf.get_collection("y_pred")
        accuracy = tf.get_collection("accuracy")
        loss = tf.get_collection("loss")

        eval_y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        eval_accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        eval_loss = sess.run(loss, feed_dict={x: X, y: Y})
        return eval_y_pred, eval_accuracy, eval_loss
