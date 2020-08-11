#!/usr/bin/env python3
"""Viterbi algorithm usign recursive approach"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Function viterbi
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
                    Transition[i, j]: the probability of transitioning from
                    the hidden state i to j
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
                 starting in a particular hidden state
    Returns: path, P, or None, None on failure
    """
    try:
        T = Observation.shape[0]
        N, M = Emission.shape

        # backpointer initialization
        backpointer = np.zeros((N, T))

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
                F[n, t] = np.amax(Transitions * F[:, t - 1]
                                  * Emissions)
                backpointer[n, t - 1] = np.argmax(Transitions * F[:, t - 1]
                                                  * Emissions)

        # Path Array
        path = [0 for i in range(T)]
        # Find the most probable last hidden state
        last_state = np.argmax(F[:, T - 1])
        path[0] = last_state

        # formula shorturl.at/uvAPU
        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            path[backtrack_index] = int(backpointer[int(last_state), i])
            last_state = backpointer[int(last_state), i]
            backtrack_index += 1

        # Flip the path array using reverse to maintain main structure
        path.reverse()

        P = np.amax(F, axis=0)
        P = np.amin(P)
        return path, P
    except Exception:
        None, None
