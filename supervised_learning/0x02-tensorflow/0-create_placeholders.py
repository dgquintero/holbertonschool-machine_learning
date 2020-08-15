#!/usr/bin/env python3
"""Placeholders function"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Arguments:
        nx: number of feature columns in our data
        classes: the number of classes in our classifier
    Return: placeholders x and y
        x: input data to the NN
        y: one-hot labels for the input data
    """
    x = tf.placeholder("float", shape=[None, nx], name='x')
    y = tf.placeholder("float", shape=[None, classes], name='y')
    return x, y
