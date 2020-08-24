#!/usr/bin/env python3
""" learning rate decay hyperparameter"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ updates the learning rate using inverse time decay
    Argumnets:
        alpha: is the original learning rate
        decay_rate: is the weight used to determine the rate
            at which alpha will decay
        global_step: is the number of passes of gradient descent
            that have elapsed
        decay_step: is the number of passes of gradient descent
            that should occur before alpha is decayed
    """
    alpha = tf.train.inverse_time_decay(alpha,
                                        global_step,
                                        decay_step,
                                        decay_rate,
                                        staircase=True)
    return alpha
