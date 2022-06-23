import json
import torch
from agent import Agent
import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser(description='Env select')
parser.add_argument('-env', type=str, help='lunar / mount / ant / cheetah / hopper / walker / humanoid',
                    choices=['lunar', 'mount', 'cheetah', 'ant', 'hopper', 'walker', 'humanoid'])

parser.add_argument('-seed', type=int, help='enter seed value')

args = parser.parse_args()

if args.env == 'lunar':
    print('LunarLander environment selected')
    env = gym.make('LunarLander-v2')

elif args.env == 'mount':
    print('MountainCar environment selected')
    env = gym.make('MountainCar-v0')

elif args.env == 'ant':
    print('Ant environment selected')
    env = gym.make('Ant-v2')

elif args.env == 'cheetah':
    print('HalfCheetah environment selected')
    env = gym.make('HalfCheetah-v2')

elif args.env == 'hopper':
    print('Hopper environment selected')
    env = gym.make('Hopper-v2')

elif args.env == 'walker':
    print('Walker environment selected')
    env = gym.make('Walker2d-v2')

elif args.env == 'humanoid':
    print('Humanoid environment selected')
    env = gym.make('Humanoid-v2')

# env.seed(0)
print(f'Seed value: {args.seed}')
env.seed(args.seed)

print('State size: ', env.observation_space.shape[0])
print('Action size: ', env.action_space.shape[0])

#agent = Agent(state_size=env.observation_space.shape[0],
#              action_size=env.action_space.n, seed=0)

agent = Agent(state_size = env.observation_space.shape[0], action_size = env.action_space.shape[0], seed=0)

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
            # time.sleep(0.2)
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            # time.sleep(0.2)
            # if (env.drone_y == env.man_y):
            #     print('\nReward: {:.4f}'.format(reward))
            #print(f'\nStates : {state}')
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
                       'checkpoint_d2qn_'+str(args.env)+'_'+str(args.seed)+'.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
    return scores


start_time = time.time()
scores_ddqn = train_agent(agent, env)
end_time = time.time()

scores_ddqn_np = np.array(scores_ddqn)
np.savetxt("scores_d2qn_"+str(args.env)+"_"+str(args.seed)+".txt", scores_ddqn_np)


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


train_info_dictionary = {'algorithm': 'D2QN', 'env': args.env, 'eps_start': eps_start, 'eps_end': eps_end,
                         'eps_decay': eps_decay, 'episodes': n_episodes, 'train_time': train_time}

train_info_file = open('train_info_'+str(args.env)+'_'+str(args.seed)+'.json', 'w')
json.dump(train_info_dictionary, train_info_file)
train_info_file.close()


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


scores_ma_ddqn = moving_average(scores_ddqn, n=100)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_ma_ddqn)), scores_ma_ddqn)
plt.ylabel('Score')
plt.xlabel('Episode')
plt.savefig('graph_'+str(args.env)+'_'+str(args.seed)+'.pdf')
plt.show()
