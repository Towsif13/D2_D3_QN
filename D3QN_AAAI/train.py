import json
import time
import torch
from agent import Agent
import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Env select')
parser.add_argument('-env', type=str, help='lunar / mount',
                    choices=['lunar', 'mount', 'cheetah'])
args = parser.parse_args()

if args.env == 'lunar':
    print('LunarLander environment selected')
    env = gym.make('LunarLander-v2')
elif args.env == 'mount':
    print('MountainCar environment selected')
    env = gym.make('MountainCar-v0')
elif args.env == 'cheetah':
    print('HalfCheetah environment selected')
    env = gym.make('Ant-v2')


env.seed(0)
agent = Agent(state_size = 17, action_size = 6, seed=0)


# parser = argparse.ArgumentParser(description='Env select')
# parser.add_argument('-env', type=str, help='lunar / mount',
#                     choices=['lunar', 'mount'])
# args = parser.parse_args()

# if args.env == 'lunar':
#     print('LunarLander environment selected')
#     env = gym.make('LunarLander-v2')
# elif args.env == 'mount':
#     print('MountainCar environment selected')
#     env = gym.make('MountainCar-v0')


# env.seed(0)
print(env.action_space[1])
# print(env.observation_space.shape[0])
# agent = Agent(state_size=env.observation_space.shape[0],
#               action_size=env.action_space.n, seed=0)

n_episodes = 10_000
max_t = 200
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.9995


def train_agent(agent, env, eps_start=eps_start, eps_decay=eps_decay, eps_end=eps_end, n_episodes=n_episodes, max_t=max_t):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_score = -150.0
    for i_episode in range(1, n_episodes+1):

        state = env.reset()
        score = 0
        for t in range(max_t):
            # env.render()
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        # if i_episode % 10 == 0:
        #     torch.save(agent.qnetwork_local.state_dict(),
        #                "dqn_agent{}.pkl".format(i_episode))
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= max_score:
            max_score = np.mean(scores_window)
            torch.save(agent.qnetwork_local.state_dict(),
                       'checkpoint_d3qn_'+str(args.env)+'.pth')
            # print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
            #     i_episode-100, np.mean(scores_window)))
    return scores


start_time = time.time()
scores_dddqn = train_agent(agent, env)
end_time = time.time()

scores_dddqn_np = np.array(scores_dddqn)
np.savetxt("scores_d3qn_"+str(args.env)+".txt", scores_dddqn_np)


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "Execution time: %d hours : %02d minutes : %02d seconds" % (hour, minutes, seconds)


n = end_time-start_time
train_time = convert(n)
print(train_time)


train_info_dictionary = {'algorithm': 'D3QN', 'env': args.env, 'eps_start': eps_start, 'eps_end': eps_end,
                         'eps_decay': eps_decay, 'episodes': n_episodes, 'train_time': train_time}

train_info_file = open('train_info_'+str(args.env)+'.json', 'w')
json.dump(train_info_dictionary, train_info_file)
train_info_file.close()


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


scores_ma_dddqn = moving_average(scores_dddqn, n=100)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_ma_dddqn)), scores_ma_dddqn)
plt.ylabel('Score')
plt.xlabel('Episode')
plt.savefig('graph_'+str(args.env)+'.png')
plt.show()
