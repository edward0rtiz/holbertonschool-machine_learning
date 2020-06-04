#!/usr/bin/env python3
""" Script that perfoms a valid convolution"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function to perform a graysclae convolution
    Args:
        images: numpy.ndarray with shape (m, h, w)
                containing multiple grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing
                the kernel for the convolution
                kn: the height of the kernel
                kw: the width of the kernel
    Returns: numpy.ndarray containing the convolved images

    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    output_h = h - kh + 1
    output_w = w - kw + 1

    # convolution output
    conv_out = np.zeros((m, output_h, output_w))

    image = np.arange(m)
    # Loop every pixel of the output
    for x in range(output_h):
        for y in range(output_w):
            # element wise multiplication of the kernel and the image
            conv_out[image, x, y] = (np.sum(images[image, x:kh+x,
                                            y:kw+y] * kernel, axis=(1, 2)))
    return conv_out
