#!/usr/bin/env python3
""" pool_forward function"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Implements the forward pass of the pooling layer
    Arguments:
        dA: (m, h_new, w_new, c_new) containing the partial derivatives
            m: number of examples
            h_new: height of the output
            w_new: width of the output
            c: number of channels
        A_prev: (m, h_prev, w_prev, c_prev) output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        kernel_shape: (kh, kw) size of the kernel for the pooling
            kh: kernel height
            kw: kernel width
        stride: tuple of (sh, sw) strides for the convolution
            sh: stride for the height
            sw: stride for the width
        mode: string containing either max or avg (maximun o avg pooling)
    Returns: output the pooling layer
    """
    # Retrieve dimensions from A_prev's shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    # Retrieve dimensions from dA's shape
    (m, h_new, w_new, c_new) = dA.shape
    # Retrieve information from "kernel_shape"
    kh, kw = kernel_shape
    # Retrieve information from "stride"
    sh, sw = stride
    # Initialize dA_prev
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    # compute the poolong operation
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end,
                                              horiz_start:horiz_end, c]
                        # mask from a prev_slice mask = x == np.max(x)
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # define the (3D) slice of a_prev_pad
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end,
                                c] += np.multiply(mask, dA[i, h, w, c])
                    elif mode == "avg":
                        da = dA[i, h, w, c]
                        shape = kernel_shape
                        average = da / (kh * kw)
                        Z = np.ones(shape) * average
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += Z
    return dA_prev
