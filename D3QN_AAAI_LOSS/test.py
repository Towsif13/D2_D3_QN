from agent import Agent
import gym
import random
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from environment import Environment
import time

#env = gym.make('LunarLander-v2')
#env = gym.make('CartPole-v0')
env = Environment()
# print('State shape: ', env.observation_space.shape)
# print('Number of actions: ', env.action_space.n)


agent = Agent(state_size=2, action_size=4, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_d3qn.pth'))

for i in range(10):
    state = env.reset()
    for j in range(200):
        action = agent.act(state)

        state, reward, done = env.step(action)
        env.render()
        time.sleep(0.4)
        if done:
            time.sleep(0.4)
            break
