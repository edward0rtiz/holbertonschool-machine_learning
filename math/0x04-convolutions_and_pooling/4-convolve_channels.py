#!/usr/bin/env python3
"""Script to perform a convolution of images with channels"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Function to perform a convolution of img with channels
    Args:
        images: numpy.ndarray with shape (m, h, w) containing
                multiple grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw, c) containing
                the kernel for the convolution
                kn: the height of the kernel
                kw: the width of the kernel
                c:  the number of channels in the image
        padding: is either a tuple of (ph, pw), 'same, 'valid'
                 ph: is the padding for the height of the image
                 pw: is the padding for the width of the image
        stride: is a tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image
    Returns: numpy.ndarray containing the convolved images

    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    pad_w = 0
    pad_h = 0
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        pad_h = int((((h - 1) * sh + kh - h) / 2) + 1)
        pad_w = int((((w - 1) * sw + kw - w) / 2) + 1)
    if type(padding) == tuple:
        pad_h = padding[0]
        pad_w = padding[1]

    image_pad = np.pad(images, pad_width=((0, 0), (pad_h, pad_h),
                                          (pad_w, pad_w), (0, 0)),
                       mode='constant')

    output_h = int(((h + 2 * pad_h - kh) / sh) + 1)
    output_w = int(((w + 2 * pad_w - kw) / sw) + 1)

    # convolution output
    conv_out = np.zeros((m, output_h, output_w))

    image = np.arange(m)
    # Loop every pixel of the output
    for x in range(output_h):
        for y in range(output_w):
            # element wise multiplication of the kernel and the image
            conv_out[image, x, y] = (np.sum(image_pad[image,
                                            x * sh:((x * sh) + kh),
                                            y * sw:((y * sw) + kw)] * kernel,
                                            axis=(1, 2, 3)))
    return conv_out
