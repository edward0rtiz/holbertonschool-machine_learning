#!/usr/bin/env python3
"""Script to train in tensorflow"""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations,
          save_path="/tmp/model.ckpt"):

    # tensors for input
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    # build graph collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    # build train and add to graph collection
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    # set global initializer
    init = tf.global_variables_initializer()
    # create saver
    saver = tf.train.Saver()
    # launch the graph and train saving the model every 100 ops
    sess = tf.Session()
    sess.run(init)
    for step in range(iterations + 1):
        t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        t_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        v_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        if step % 100 == 0:
            print("After {} iterations:".format(step))
            print("\tTraining Cost:".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_acc))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_acc))
        if step < iterations:
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
    return saver.save(sess, save_path)
