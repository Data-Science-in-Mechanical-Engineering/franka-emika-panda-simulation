import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
plt.rcParams.update({'font.size': 16})
os.chdir(os.path.dirname(os.path.realpath(__file__)))
Rewards_gosafe=pd.read_csv("contextual/gosafe/Gosafe_Reward.csv",header=None)
Rewards_safeopt=pd.read_csv("contextual/SafeOpt/SafeOpt_Reward.csv", header=None)
Rewards_eic=pd.read_csv("contextual/EIC/eic_Reward.csv", header=None)
Rewards_gosafe=np.asarray(Rewards_gosafe)
Rewards_safeopt=np.asarray(Rewards_safeopt)
Rewards_eic=np.asarray(Rewards_eic)

x=np.linspace(0,200,41)
mean_rewards_gosafe=np.mean(Rewards_gosafe,axis=1)
mean_rewards_safeopt=np.mean(Rewards_safeopt,axis=1)
mean_rewards_eic=np.mean(Rewards_eic,axis=1)
plot_type=2
if plot_type==1:
    fig = plt.figure(figsize=(10, 8))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    ax.plot(x, mean_rewards_safeopt, label = "SafeOptSwarm", linestyle="-.",c="g")
    ax.scatter(x, mean_rewards_safeopt,c="g")
    ax.errorbar(x,mean_rewards_safeopt,yerr=np.std(Rewards_safeopt,axis=1)/np.sqrt(Rewards_safeopt.shape[1]),fmt=".k", lw=2, capsize=2, capthick=2,ecolor="g")
    #ax.plot(x, mean_rewards_eic, label = "EIC", linestyle="-.",c="r")
    #ax.scatter(x, mean_rewards_eic,c="r")
    #ax.errorbar(x,mean_rewards_eic,yerr=np.std(Rewards_eic,axis=1)/np.sqrt(Rewards_eic.shape[1]),fmt=".k", lw=2, capsize=2, capthick=2,ecolor="r")
    ax.plot(x, mean_rewards_gosafe, label = "Contextual GoSafe", linestyle="-.",c="b")
    ax.scatter(x, mean_rewards_gosafe,c="b")
    ax.scatter(100,mean_rewards_gosafe[20],c='gold',marker='x',s=200,linewidths=5,label='Jump')
    ax.errorbar(x,mean_rewards_gosafe,yerr=np.std(Rewards_gosafe,axis=1)/np.sqrt(Rewards_gosafe.shape[1]),fmt=".k", lw=2, capsize=2, capthick=2,ecolor="b")


    ax.set_ylim([-6, 2])


    plt.xlim([-1, 201])
    ax.set_xlabel('Number of Experiments')
    ax.set_ylabel('Normalized Rewards')
    ax.legend()
    #plt.xticks(np.arange(0, 200, 41))
    plt.savefig("SafeOptGoSafe_ArmSimcomparison.png",dpi=300)
else:
    methods = ['SafeOptSwarm', 'EIC', 'Contextual GoSafe']
    x_pos = np.arange(len(methods))
    baseline = -2
    CTEs = [mean_rewards_safeopt[40]-baseline, mean_rewards_eic[40]-baseline, mean_rewards_gosafe[40]-baseline]
    error = [np.std(Rewards_safeopt[40,:])/np.sqrt(Rewards_safeopt.shape[1]), np.std(Rewards_eic[40,:])/np.sqrt(Rewards_eic.shape[1]), np.std(Rewards_gosafe[40,:])/np.sqrt(Rewards_gosafe.shape[1])]
    fig = plt.figure(figsize=(10, 8))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    #fig, ax = plt.subplots()

    bar = ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.9,color=['orchid', 'cadetblue', 'salmon'], capsize=10,bottom=baseline)
    ax.set_ylabel('Normalized Rewards')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)

    ax.yaxis.grid(True)
    sterr=[0,15.6,0]
    c=['green','red','green']
    #scale=[0.1,0.48,0.48]
    for i, rectangle in enumerate(bar):
        height = rectangle.get_height()
        plt.text(rectangle.get_x() + rectangle.get_width() / 2, height+baseline + error[i]+ 0.01,
                  (str(sterr[i])),
                 ha='center', va='bottom',color=c[i])
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plots_comparison.png',dpi=300)

