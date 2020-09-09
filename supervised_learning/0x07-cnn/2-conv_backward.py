#!/usr/bin/env python3
""" conv_forward function"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Funtion that performs forward prop over a conv layer of NN
    Arguments:
        dZ: (m, h_new, w_new, c_new) partial derivatives with
            respect to the unactivated output of the convolutional layer
            m: number of examples
            h_new: height of the output
            w_new: width of the output
            c_new: number of channels in the output
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
    # Retrieve dimensions from dZ's shape
    (m, h_new, w_new, c_new) = dZ.shape

    if padding is 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros((1, 1, 1, c_new))

    # create an image Pad A_prev and dA_prev
    pad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_prev_pad = np.pad(A_prev, pad_width=pad, mode='constant')
    dA_prev_pad = np.pad(dA_prev, pad_width=pad, mode='constant')

    for i in range(m):
        # select the training example from A_prev ans da_prev
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    # Use the corners to define the (3D) slice of a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end]
                    # update gradients for the window and the filter
                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end] += W[:, :, :, c] *\
                        dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    # db[:, :, :, c] += dZ[i, h, w, c]
    # set the
        if padding == 'same':
            dA_prev[i, :, :, :] += da_prev_pad[ph:-ph, pw:-pw, :]
        if padding == 'valid':
            dA_prev[i, :, :, :] += da_prev_pad

    return dA_prev, dW, db
