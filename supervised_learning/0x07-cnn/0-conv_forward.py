#!/usr/bin/env python3
""" Script to forward propagate over a convolutional layer in a NN"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1, 1)):
    """
    Function to forward propagate over a convolutional layer in a NN
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
                m: is the number of examples
                h_prev: the height of the previous layer
                w_prev: the width of the previous layer
                c_prev: the number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution
            kh: the filter height
            kw: the filter width
            c_prev: the number of channels in the previous layer
            c_new: the number of channels in the output
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution
        activation: an activation function applied to the convolution
        padding: string that is either same or valid, indicating the type of
                 padding used
        stride: tuple of (sh, sw) containing the strides for the convolution
                sh: the stride for the height
                sw: the stride for the width
    Returns: the output of the convolutional layer
    """

    # Retrieve the dimensions from A_prev shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieve the dimensions from A_prev shape
    (kh, kw, c_prev, c_new) = W.shape

    # Retrieve the values of the stride
    sh, sw = stride

    # Padding values
    pw = 0
    ph = 0

    if padding == 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    if padding == 'valid':
        ph = 0
        pw = 0

    # Create an image pad using np.pad
    img_pad = np.pad(A_prev,
                     pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant')

    # Compute the dimensions of the CONV output volume
    c_h = int(((h_prev + 2 * ph - kh) / sh) + 1)
    c_w = int(((w_prev + 2 * pw - kw) / sw) + 1)

    # Initialize the output volume conv (Z) with zeros
    conv = np.zeros((m, c_h, c_w, c_new))

    # Loop over the vertical_ax, then horizontal_ax, then over channel
    for i in range(c_h):
        for j in range(c_w):
            for k in range(c_new):
                # Use corners to define 3D slice of img_pad element wise
                # sth = i * sh // endh =  ((i * sh) + kh)
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw
                img_slice = img_pad[:, v_start:v_end, h_start:h_end]
                kernel = W[:, :, :, k]
                conv[:, i, j, k] = (np.sum(np.multiply(img_slice,
                                                       kernel),
                                                       axis=(1, 2, 3)))
    Z = conv + b
    return activation(Z)
