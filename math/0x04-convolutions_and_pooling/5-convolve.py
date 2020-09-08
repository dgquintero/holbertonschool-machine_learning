#!/usr/bin/env python3
""" convolve function"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """that performs a convolution on images using multiple kernels
    Arguments:
        images: shape (m, h, w, c) containing multiple grayscale images
            m: the number of images
            h: the height in pixels of the images
            w: the width in pixels of the images
            c: is the number of channels in the image
        kernel: shape (kh, kw, c, nc) containing the kernel for the convolution
            kh: the height of the kernel
            kw: the width of the kernel
            nc: is the number of kernels
        padding: is a tuple of (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image
        stride:
            sh: stride for the height of the image
            sw: stride for the width of the image
    Returns: ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    c = kernels.shape[2]
    nc = kernels.shape[3]
    padh = 0
    padw = 0
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        padh = int((((h - 1) * sh + kh - h) / 2) + 1)
        padw = int((((w - 1) * sw + kw - w) / 2) + 1)

    if type(padding) == tuple:
        padh = padding[0]
        padw = padding[1]

    pad = ((0, 0), (padh, padh), (padw, padw), (0, 0))
    new_h = int(((h + (2 * padh) - kh) / sh) + 1)
    new_w = int(((w + (2 * padw) - kh) / sw) + 1)

    image_p = np.pad(images, pad_width=pad, mode='constant')

    output_c = np.zeros((m, new_h, new_w, nc))

    for i in range(new_h):
        for j in range(new_w):
            for k in range(nc):
                x = i * sh
                y = j * sw
                image = image_p[:, x:x+kh, y:y+kw, :]
                output_c[:, i, j, k] = np.multiply(image,
                                                   kernels[:, :, :, k]).\
                    sum(axis=(1, 2, 3))
    return output_c
