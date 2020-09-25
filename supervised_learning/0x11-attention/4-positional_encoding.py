#!/usr/bin/env python3

import numpy as np

def positional_encoding(max_seq_len, dm):
    """

    Args:
        max_seq_len:
        dm:

    Returns:
    """
    p_encoding = np.zeros([max_seq_len, dm])

    for i in range(dm):
        for p in range(max_seq_len):
            p_encoding[p, i] = p / np.power(1000, (2 * (i // 2) / dm))

        p_encoding[:, 0::2] = np.sin(p_encoding[:, 0::2])
        p_encoding[:, 1::2] = np.cos(p_encoding[:, 1::2])
    return p_encoding