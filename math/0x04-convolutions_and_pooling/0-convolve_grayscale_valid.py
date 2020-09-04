#!/usr/bin/env python3
""" convolve_grayscale_valid function"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """function that performs a valid convolution on grayscale images
    Arguments:
        images: shape (m, h, w) containing multiple grayscale images
            m: the number of images
            h: the height in pixels of the images
            w: the width in pixels of the images
        kernel: shape (kh, kw) containing the kernel for the convolution
            kh: the height of the kernel
            kw: the width of the kernel
    Returns: ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    output_h = h - kh + 1
    output_w = w - kw + 1

    # convolution output
    output_conv = np.zeros((m, output_h, output_w))

    image = np.arange(m)

    for x in range(output_h):
        for y in range(output_w):
            output_conv[image, x, y] = np.sum(images[image, x:kh+x,
                                              y:kw+y] * kernel, axis=(1, 2))

    return output_conv
