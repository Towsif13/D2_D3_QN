import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


scores_d2qn = np.loadtxt("scores_d2qn.txt")
scores_ma_d2qn = moving_average(scores_d2qn, n=100)
scores_ma_d2qn_1000 = moving_average(scores_d2qn, n=1000)

plt.plot(np.arange(len(scores_ma_d2qn)), scores_ma_d2qn,
         alpha=0.2, color='b')
plt.plot(np.arange(len(scores_ma_d2qn_1000)),
         scores_ma_d2qn_1000, label="D2QN", color='b')

plt.ylabel('Score')
plt.xlabel('Episode ')
plt.legend()
plt.savefig('graph.png')
plt.show()
