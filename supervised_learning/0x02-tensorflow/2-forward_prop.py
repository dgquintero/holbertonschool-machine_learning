#!/usr/bin/env python3
"""forward_prop function"""


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    Arguments:
        x: the placeholder for input data
        layer_sizes: lis containing the number of nodes in each layer
        activation: list containing the activation function for each layer
    Returns: the prediction of the network in tensor form
    """
    layer = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
