#!/usr/bin/env python3
""" Forward Markov chain"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Function that performs the forward algorithm of a HMM
    Args:
        Observation: numpy.ndarray of shape (T,) that contains the index
                     of the observation
                     T: the number of observations
        Emission: numpy.ndarray of shape (N, M) containing the emission
                  probability of a specific observation given a hidden state
                  Emission[i, j]: the probability of observing j given the
                                  hidden state i
                  N: the number of hidden states
                  M: the number of all possible observations
        Transition: 2D numpy.ndarray of shape (N, N) containing the transition
                    probabilities
                    Transition[i, j] is the probability of transitioning from
                    the hidden state i to j
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
                 starting in a particular hidden state
    Returns: P, F, or None, None on failure
             P: the likelihood of the observations given the model
             F: numpy.ndarray of shape (N, T) containing the forward path
                probabilities
                F[i, j]: the probability of being in hidden state i at time
                         j given the previous observations
    """
    try:
        # Hidden States
        N = Transition.shape[0]

        # Observations
        T = Observation.shape[0]

        # F == alpha
        # initialization α1(j) = πjbj(o1) 1 ≤ j ≤ N
        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        # formula shorturl.at/amtJT
        # Recursion αt(j) == ∑Ni=1 αt−1(i)ai jbj(ot); 1≤j≤N,1<t≤T
        for t in range(1, T):
            for n in range(N):
                Transitions = Transition[:, n]
                Emissions = Emission[n, Observation[t]]
                F[n, t] = np.sum(Transitions * F[:, t - 1]
                                 * Emissions)

        # Termination P(O|λ) == ∑Ni=1 αT (i)
        P = np.sum(F[:, -1])
        return P, F
    except Exception:
        None, None
