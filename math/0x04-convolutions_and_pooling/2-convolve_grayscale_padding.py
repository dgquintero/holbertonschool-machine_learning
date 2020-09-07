#!/usr/bin/env python3
""" convolve_grayscale_same function"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """function that performs a valid convolution on grayscale images
    Arguments:
        images: shape (m, h, w) containing multiple grayscale images
            m: the number of images
            h: the height in pixels of the images
            w: the width in pixels of the images
        kernel: shape (kh, kw) containing the kernel for the convolution
            kh: the height of the kernel
            kw: the width of the kernel
        padding: is a tuple of (ph, pw)
    Returns: ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    padh = padding[0]
    padw = padding[1]
    pad = ((0, 0), (padh, padh), (padw, padw))
    new_h = h + (2 * padh) - kh + 1
    new_w = w + (2 * padw) - kw + 1

    image_p = np.pad(images, pad_width=pad, mode='constant')

    output_conv = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            image = image_p[:, i:i+kh, j:j+kw]
            output_conv[:, i, j] = np.multiply(image, kernel).sum(axis=(1,2))

    return output_conv
