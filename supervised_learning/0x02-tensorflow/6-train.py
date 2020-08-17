#!/usr/bin/env python3
"""create_train_op function"""
import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network classifier
    Arguments:
        X_train: training input data
        Y_train: containing the training labels
        X_valid: containing the validation input data
        Y_valid: containing the validation labels
        layer_sizes: list the number of nodes in each layer of the network
        activations: list activation functions for each layer of the network
        alpha: learning rate
        iterations: number of iterations to train over
        save_path: designates where to save the model
    Returns: the path where the model was saved
    """
    # create placeholders for input
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    # graph's collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    # train fuction
    train_opti = create_train_op(loss, alpha)
    tf.add_to_collection('train_opti', train_opti)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(iterations + 1):
            t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            if epoch % 100 == 0 or epoch == iterations:
                print("After {} iterations:".format(epoch))
                print("\tTraining Cost: {}".format(t_cost))
                print("\tTraining Accuracy: {}".format(t_accuracy))
                print("\tValidation Cost: {}".format(v_cost))
                print("\tValidation Accuracy: {}".format(v_accuracy))
            if epoch < iterations:
                sess.run(train_opti, feed_dict={x: X_train, y: Y_train})
        return saver.save(sess, save_path)
