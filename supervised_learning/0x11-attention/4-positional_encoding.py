#!/usr/bin/env python3
""" postitional encoding"""

import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    """Get angles"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    """
    positional encoding
    Args:
        max_seq_len: Integer representing the maximum sequence
        dm: model depth
    Returns: numpy.ndarray of shape (max_seq_len, dm) containing
             the positional encoding vectors
    """

    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :],
                            dm)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
