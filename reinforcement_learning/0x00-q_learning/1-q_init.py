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
    action = env.action_space
    space = env.observation_space.n

    q_table = np.zeros((action, space))

    return q_table
