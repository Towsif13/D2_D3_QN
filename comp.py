import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


#plt.figure(figsize=(15, 8))
plt.figure(figsize=(7, 5))


################################ DQN TARGET #######################################
dqn_mount = np.loadtxt("D2QN_AAAI\scores_d2qn_cheetah_0.txt")
ma_100_dqn_mount = moving_average(dqn_mount, n=10)
ma_1000_dqn_mount = moving_average(dqn_mount, n=100)

################################ LOSS STRATEGY #######################################
loss_mount = np.loadtxt("D2QN_AAAI_LOSS\scores_d2qn_loss_r-50_cheetah_0.txt")
ma_100_loss_mount = moving_average(loss_mount, n=10)
ma_1000_loss_mount = moving_average(loss_mount, n=100)

################################ REWARD STRATEGY #######################################
reward_mount = np.loadtxt(
    "D2QN_AAAI_REWARD\scores_d2qn_reward-50_cheetah_0.txt")
ma_100_reward_mount = moving_average(reward_mount, n=10)
ma_1000_reward_mount = moving_average(reward_mount, n=100)

#################################### PLOTS ##############################################

plt_1_ma_100 = ma_100_dqn_mount
plt_1_ma_1000 = ma_1000_dqn_mount

plt_2_ma_100 = ma_100_loss_mount
plt_2_ma_1000 = ma_1000_loss_mount

plt_3_ma_100 = ma_100_reward_mount
plt_3_ma_1000 = ma_1000_reward_mount


plt.plot(np.arange(len(plt_1_ma_100)), plt_1_ma_100, alpha=0.1, color='r')
plt.plot(np.arange(len(plt_1_ma_1000)), plt_1_ma_1000,
         alpha=1, color='r', label="DQN")

plt.plot(np.arange(len(plt_2_ma_100)), plt_2_ma_100, alpha=0.1, color='b')
plt.plot(np.arange(len(plt_2_ma_1000)), plt_2_ma_1000,
         alpha=1, color='b', label="LSS, $\u03C4$ = 10")

plt.plot(np.arange(len(plt_3_ma_100)), plt_3_ma_100, alpha=0.1, color='g')
plt.plot(np.arange(len(plt_3_ma_1000)), plt_3_ma_1000,
         alpha=1, color='g', label="RSS, $\u03C4$ = -20")


plt.xlim(xmin=0, xmax=10_051)
# plt.ylim(ymin=-335)

plt.ylabel('Reward', fontsize=15)
plt.xlabel('Episode', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.title('All DQN comparison', fontsize=15)
plt.legend(facecolor='white', fontsize=15)
plt.savefig('comparison_graph_cheetah.pdf', bbox_inches='tight', dpi=350)
plt.savefig('comparison_graph_cheetah.png', bbox_inches='tight', dpi=350)
plt.show()
