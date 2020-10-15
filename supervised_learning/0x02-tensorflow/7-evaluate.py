#!/usr/bin/env python3
"""evaluate function"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    evaluates the output of a neural network
    Arguments:
        X: the input data to evaluate
        Y: the one-hot labels for X
        save_path: location to load the model from
    Returns: the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(sess, save_path)
        #  tf.get_collection() returns a list. Retrieve only the 1st one
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        eval_y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        eval_accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        eval_loss = sess.run(loss, feed_dict={x: X, y: Y})
        return eval_y_pred, eval_accuracy, eval_loss
