#!/usr/bin/env python3
""" postitional encoding"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    positional encoding
    Args:
        max_seq_len: Integer representing the maximum sequence
        dm: model depth
    Returns: numpy.ndarray of shape (max_seq_len, dm) containing
             the positional encoding vectors
    """
    p_encoding = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    dm_n = np.float32(dm)

    grad_angle = 1 / (np.power(10000, (2 * (i // 2) / dm_n)))
    angle = p_encoding * grad_angle

    positional = np.zeros((max_seq_len, dm))

    positional[:, 0::2] = np.sin(angle[:, 0::2])
    positional[:, 1::2] = np.cos(angle[:, 1::2])
    return positional
