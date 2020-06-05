#!/usr/bin/env python3
"""Script to perform a convolution of images with multiple kernels"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function to perform a convolution of img with channels
    Args:
        images: numpy.ndarray with shape (m, h, w) containing
                multiple grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
                c: number the channels in the image
        kernel_shape: tuple of (kh, kw) containing
                the kernel shape of the pooling
                kh: the height of the kernel
                kw: the width of the kernel
        stride: is a tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image
        mode: indicates the type of pooling
                 max: max pooling
                 avg: average pooling
        Returns: numpy.ndarray containing the pooled images

    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    output_h = int(1 + ((h - kh) / sh))
    output_w = int(1 + ((w - kw) / sw))

    # convolution output
    conv_out = np.zeros((m, output_h, output_w, c))

    image = np.arange(m)

    # Loop every pixel of the output
    for x in range(output_h):
        for y in range(output_w):
            # pooling implementation
            if mode == 'max':
                conv_out[image, x, y] = (np.max(images[image,
                                                x * sh:((x * sh) + kh),
                                                y * sw:((y * sw) + kw)],
                                                axis=(1, 2)))
            elif mode == 'avg':
                conv_out[image, x, y] = (np.mean(images[image,
                                                x * sh:((x * sh) + kh),
                                                y * sw:((y * sw) + kw)],
                                                axis=(1, 2)))
    return conv_out
