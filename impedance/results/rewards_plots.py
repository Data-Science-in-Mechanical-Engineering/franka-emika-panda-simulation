import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
Rewards=pd.read_csv("SafeOpt_Reward.csv",header=None)

Rewards=np.asarray(Rewards)
fig = plt.figure(figsize=(10, 10))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
x=np.linspace(0,200,41)
mean_rewards=np.mean(Rewards,axis=1)
ax.plot(x, mean_rewards, label = "Reward Curve", linestyle="-.")
ax.scatter(x, mean_rewards)
ax.errorbar(x,mean_rewards,yerr=np.std(Rewards,axis=1),fmt=".k", lw=1, capsize=2, capthick=2)
ax.set_ylim([0, 1])
plt.xlim([-1, 201])
#plt.xticks(np.arange(0, 200, 41))
plt.show()
check=True