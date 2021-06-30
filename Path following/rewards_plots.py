import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
plt.rcParams.update({'font.size': 16})
os.chdir(os.path.dirname(os.path.realpath(__file__)))
Rewards_gosafe=pd.read_csv("gosafe/20_seeds/Gosafe_Reward.csv",header=None)
Rewards_safeopt=pd.read_csv("safeopt/20_seeds/SafeOpt_Reward.csv", header=None)

Rewards_gosafe=np.asarray(Rewards_gosafe)
Rewards_safeopt=np.asarray(Rewards_safeopt)
fig = plt.figure(figsize=(10, 8))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
x=np.linspace(0,200,41)
mean_rewards_gosafe=np.mean(Rewards_gosafe,axis=1)
mean_rewards_safeopt=np.mean(Rewards_safeopt,axis=1)
ax.plot(x, mean_rewards_gosafe, label = "GoSafeSwarm", linestyle="-.",c="b")
ax.scatter(x, mean_rewards_gosafe,c="b")
ax.errorbar(x,mean_rewards_gosafe,yerr=np.std(Rewards_gosafe,axis=1)/np.sqrt(Rewards_gosafe.shape[1]),fmt=".k", lw=2, capsize=2, capthick=2,ecolor="b")

ax.plot(x, mean_rewards_safeopt, label = "SafeOptSwarm", linestyle="-.",c="g")
ax.scatter(x, mean_rewards_safeopt,c="g")
ax.errorbar(x,mean_rewards_safeopt,yerr=np.std(Rewards_safeopt,axis=1)/np.sqrt(Rewards_safeopt.shape[1]),fmt=".k", lw=2, capsize=2, capthick=2,ecolor="g")
ax.set_ylim([-6, 2])
plt.xlim([-1, 201])
ax.set_xlabel('Number of Experiments')
ax.set_ylabel('Normalized Rewards')
ax.legend()
#plt.xticks(np.arange(0, 200, 41))
plt.savefig("SafeOptGoSafe_ArmSimcomparison.png",dpi=300)