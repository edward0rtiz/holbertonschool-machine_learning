#!/usr/bin/env python3
""" backward algorithm"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Backward function
    Args:
        Observation: numpy.ndarray of shape (T,) that contains the index
                     of the observation
                     T: the number of observations
        Emission: numpy.ndarray of shape (N, M) containing the emission
                  probability of a specific observation given a hidden state
                  Emission[i, j]: probability of observing j given the hidden
                                  state i
                  N: the number of hidden states
                  M: the number of all possible observations
        Transition: 2D numpy.ndarray of shape (N, N) containing the transition
                    probabilities
                    Transition[i, j]: the probability of transitioning from the
                                      hidden state i to j
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
                 starting in a particular hidden state
    Returns: P, B, or None, None on failure
             P: the likelihood of the observations given the model
             B: numpy.ndarray of shape (N, T) containing the backward path
                probabilities
    """
    try:
        T = Observation.shape[0]
        N, M = Emission.shape
        beta = np.zeros((N, T))
        beta[:, T - 1] = np.ones((N))

        for t in range(T - 2, -1, -1):
            for n in range(N):
                Transitions = Transition[n, :]
                Emissions = Emission[:, Observation[t + 1]]
                beta[n, t] = np.sum((Transitions * beta[:, t + 1]) * Emissions)

        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])
        return P, beta
    except  Exception:
        return None, None