#!/usr/bin/env python3
""" convolve_grayscale function"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """that performs a convolution on images with channels
    Arguments:
        images: shape (m, h, w) containing multiple grayscale images
            m: the number of images
            h: the height in pixels of the images
            w: the width in pixels of the images
        kernel: shape (kh, kw) containing the kernel for the convolution
            kh: the height of the kernel
            kw: the width of the kernel
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
    kh = kernel.shape[0]
    kw = kernel.shape[1]
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

    output_conv = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            x = i * sh
            y = j * sw
            image = image_p[:, x:x+kh, y:y+kw, :]
            output_conv[:, i, j] = np.multiply(image, kernel).sum(axis=(1, 2, 3))

    return output_conv
