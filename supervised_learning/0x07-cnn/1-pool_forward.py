#!/usr/bin/env python3
""" pool_forward function"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ 
    Implements the forward pass of the pooling layer
    Arguments:
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

    # Retrieve information from "kernel_shape"
    kh, kw = kernel_shape

    # Retrieve information from "stride"
    sh, sw = stride

    # Compute the dimensions of the CONV output volume
    c_h = int(1 + (h_prev - kh) / sh)
    c_w = int(1 + (w_prev - kw) / sw)

    # Initialize the output volume Z with zeros.
    conv = np.zeros((m, c_h, c_w, c_prev))

    for h in range(c_h):
        for w in range(c_w):
            # Find the corners of the current "slice" (≈4 lines)
            vert_start = h * sh
            vert_end = vert_start + kh
            horiz_start = w * sw
            horiz_end = horiz_start + kw


            # Use the corners to define the (3D) slice of a_prev_pad
            img_slice = A_prev[:, vert_start:vert_end, horiz_start:horiz_end]
            # compute the poolong operation
            if mode == "max":
                conv[:, h, w] = np.max(img_slice, axis=(1, 2))
            elif mode == "average":
                conv[:, h, w] = np.mean(img_slice, axis=(1, 2))

    return conv
