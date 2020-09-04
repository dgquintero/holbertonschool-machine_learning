#!/usr/bin/env python3
""" convolve_grayscale_same function"""
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    padh = int((kh - 1) / 2)
    padw = int((kw - 1) / 2)

    if kh % 2 == 0:
        padh = int(kh / 2)
    if kw % 2 == 0:
        padw = int(kw / 2)

    image_p = np.pad(images, pad_width=((0, 0),
                                        (padh, padh),
                                        (padw, padw)),
                     mode='constant')

    output_conv = np.zeros((m, h, w))

    image = np.arange(m)

    for x in range(h):
        for y in range(w):
            output_conv[image, x, y] = np.sum(image_p[image, x:kh+x,
                                              y:kw+y] * kernel, axis=(1, 2))

    return output_conv
