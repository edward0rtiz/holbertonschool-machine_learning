#!/usr/bin/env python3
from PIL import Image
import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import tensorflow.keras as K


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

build_model = __import__('train').build_model
AtariProcessor = __import__('train').AtariProcessor


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n
    model = build_model(num_action)  # deep conv net
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    dqn = DQNAgent(model=model, nb_actions=num_actions,
                   processor=processor, memory=memory)
    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    # load weights.
    dqn.load_weights('policy.h5')

    # evaluate algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=True)