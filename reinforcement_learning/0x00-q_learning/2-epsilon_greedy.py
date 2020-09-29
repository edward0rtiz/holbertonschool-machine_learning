#!/usr/bin/env python3
"""epsilon greedy"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon greedy
    Args:
        Q: numpy.ndarray containing the q-table
        state: is the current state
        epsilon: is the epsilon to use for the calculation
    Returns: the next action index
    """

    e_tradeoff = np.random.uniform(0, 1)
    if e_tradeoff < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action
