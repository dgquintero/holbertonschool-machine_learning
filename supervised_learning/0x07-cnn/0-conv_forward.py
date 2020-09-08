#!/usr/bin/env python3
""" conv_forward function"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ 
    Funtion that performs forward prop over a conv layer of NN
    Arguments:
        A_prev: (m, h_prev, w_prev, c_prev) output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        W: (kh, kw, c_prev, c_new) kernels of the convolution
            kh: filter height
            kw: filter width
            c_prev: number of channels in the previous layer
            c_new: number of channels in the output
        b: (1, 1, 1, c_new) biases applied to the convolution
        activation: activation function applied to the convolution
        padding: type of padding either same or valid
        stride: tuple of (sh, sw) strides for the convolution
            sh: stride for the height
            sw: stride for the width
    """
    # Retrieve dimensions from A_prev's shape  
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (kh, kw, c_prev, c_new) = W.shape

    # Retrieve information from "stride"
    sh, sw = stride

    # Padding
    ph = 0
    pw = 0

    if padding is 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))

    if padding == 'valid':
        ph = 0
        pw = 0
    
    # create an image pad
    pad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    img_pad = np.pad(A_prev, pad_width=pad, mode='constant')

    # Compute the dimensions of the CONV output volume
    c_h = int(((h_prev + 2 * ph - kh) / sh) + 1)
    c_w = int(((w_prev + 2 * pw - kw) / sw) + 1)

    # Initialize the output volume Z with zeros.
    conv = np.zeros((m, c_h, c_w, c_new))

    for h in range(c_h):
        for w in range(c_w):
            for c in range(c_new):
                # Find the corners of the current "slice" (≈4 lines)
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw
                # Use the corners to define the (3D) slice of a_prev_pad
                img_slice = img_pad[:, vert_start:vert_end, horiz_start:horiz_end]
                kernel = W[:, :, :, c]
                # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                # Element-wise product between a_slice and W. Add bias.
                # Sum over all entries of the volume s
                conv[:, h, w, c] = (np.sum(np.multiply(img_slice,
                                                       kernel),
                                           axis=(1, 2, 3)))

    Z = conv + b
    return activation(Z)
