#!/usr/bin/env python3
"""Q learning"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    play
    Args:
        env: is the FrozenLakeEnv instance
        Q: umpy.ndarray containing the Q-table
        max_steps: the maximum number of steps in the episode
    Returns: total rewards for the episode
    """
    # reset the state
    state = env.reset()
    env.render()
    done = False

    for step in range(max_steps):
        # take the action with maximum expected future reward form the q-table
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)

        if done is True:
            env.render()
            return reward
        env.render()
        state = new_state

    # close the connection to the environment
    env.close()
    return reward
