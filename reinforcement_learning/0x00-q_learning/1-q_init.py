#!/usr/bin/env python3
"""Q init"""

import numpy as np


def q_init(env):
    """
    Initialize Q-table
    Args:
        env: is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """

    q_table = np.zeros((state_space_size, action_space_size))

    return q_table
