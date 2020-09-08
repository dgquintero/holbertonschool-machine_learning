#!/usr/bin/env python3
""" pool function"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """that performs a convolution on images using multiple kernels
    Arguments:
        images: shape (m, h, w, c) containing multiple grayscale images
            m: the number of images
            h: the height in pixels of the images
            w: the width in pixels of the images
            c: is the number of channels in the image
        kernel_shape: (kh, kw) containing the kernel shape for the pooling
            kh: the height of the kernel
            kw: the width of the kernel
        stride:
            sh: stride for the height of the image
            sw: stride for the width of the image
        mode: indicates the type of pooling
            max: max pooling
            avg: average pooling
    Returns: ndarray containing the pooled images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    if mode == 'max':
        pooling = np.max
    else:
        pooling = np.average

    new_h = int(((h - kh) / sh) + 1)
    new_w = int(((w - kh) / sw) + 1)

    output_c = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            x = i * sh
            y = j * sw
            output_c[:, i, j, :] = pooling(images[:, x:x+kh, y:y+kw, :],
                                           axis=(1, 2))
    return output_c
