import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({'font.size': 16})
data=pd.read_csv("data_arm_full.csv",header=None)
data_type="full"
data=np.asarray(data)

sort=np.sort(data[:,2])
p=np.percentile(sort,75)

lower_bound_g1=0.75
plot_list=["objective","constraint1","safeset","Levelset2","Levelset1"]

for plot in plot_list:
    if plot=="objective":
        fig = plt.figure(figsize=(10, 10))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        safe=np.logical_and(data[:,3]<=0.01,data[:,2]>=lower_bound_g1)

        #safe=np.logical_and(safe,data[:,0]>4)
        values=data[:,4]
        values_safe=values[safe]

        safe_max=np.where(values==values_safe.max())

        values=values.reshape([20,20])
        x=data[:,0]
        #x=np.log10(x)
        colours=['red','green']
        y=data[:,1]

        x_max=x[safe_max]
        y_max=y[safe_max]

        x=x.reshape([20,20])
        y=y.reshape([20,20])
        y_idx=np.zeros(len(y))
        #cp = plt.contour(y, x, values, inline=True)
        #ax.scatter(x, y, c=values, cmap=matplotlib.colors.ListedColormap(colours))
        #ax.clabel(cp, inline=True,
        #          fontsize=8)

        #

        ax.set_xlabel('q_r')
        ax.set_ylabel('q_d')
        cs=ax.contourf(x, y, values)

        #ax.scatter(x, y, c=values, cmap=matplotlib.colors.ListedColormap(colours))
        #ax.legend()
        #ax.scatter(x,y)
        ax.set_title("Contourplot Objective")
        ax.scatter(x_max,y_max,c="darkred")
        fig.colorbar(cs, ax=ax, shrink=0.9)
        ax.set_ylim([-0.01, 2.01])
        plt.savefig('contourplot' + data_type+'.png', dpi=300)

    elif plot=="constraint1":
        fig = plt.figure(figsize=(10, 10))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        values = data[:, 2]

        values = values.reshape([20, 20])
        x = data[:, 0]
        # x=np.log10(x)
        colours = ['red', 'green']
        y = data[:, 1]


        x = x.reshape([20, 20])
        y = y.reshape([20, 20])
        y_idx = np.zeros(len(y))

        ax.set_xlabel('q_r')
        ax.set_ylabel('q_d')
        cs = ax.contourf(x, y, values)

        ax.set_title("Contourplot Constraint 1")

        fig.colorbar(cs, ax=ax, shrink=0.9)
        ax.set_ylim([-0.01, 2.01])
        plt.savefig('contourplot_g1' + data_type+'.png', dpi=300)


    elif plot=="Levelset1":
        fig = plt.figure(figsize=(10, 10))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        values = data[:, 2]>=lower_bound_g1

        values = values.reshape([20, 20])
        x = data[:, 0]
        # x=np.log10(x)
        colours = ['red', 'green']
        y = data[:, 1]

        x = x.reshape([20, 20])
        y = y.reshape([20, 20])
        y_idx = np.zeros(len(y))

        ax.set_xlabel('q_r')
        ax.set_ylabel('q_d')
        cs = ax.contourf(x, y, values)
        ax.scatter(x, y, c=values, cmap=matplotlib.colors.ListedColormap(colours))
        ax.set_title("Levelset Constraint 1")

        #fig.colorbar(cs, ax=ax, shrink=0.9)
        ax.set_ylim([-0.01, 2.01])
        plt.savefig('Levelset_g1' + data_type +'.png', dpi=300)


    elif plot=="Levelset2":
        fig = plt.figure(figsize=(10, 10))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        values = data[:, 3]<=0.01

        values = values.reshape([20, 20])
        x = data[:, 0]
        # x=np.log10(x)
        colours = ['red', 'green']
        y = data[:, 1]

        x = x.reshape([20, 20])
        y = y.reshape([20, 20])
        y_idx = np.zeros(len(y))

        ax.set_xlabel('q_r')
        ax.set_ylabel('q_d')
        cs = ax.contourf(x, y, values)
        ax.scatter(x, y, c=values, cmap=matplotlib.colors.ListedColormap(colours))
        ax.set_title("Levelset Constraint 2")

        #fig.colorbar(cs, ax=ax, shrink=0.9)
        ax.set_ylim([-0.01, 2.01])
        plt.savefig('Levelset_g2'+ data_type+'.png', dpi=300)


    elif plot=="safeset":
        fig = plt.figure(figsize=(10, 10))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        values = np.logical_and(data[:,3]<=0.01,data[:,2]>=lower_bound_g1)

        values = values.reshape([20, 20])
        x = data[:, 0]
        # x=np.log10(x)
        colours = ['red', 'green']
        y = data[:, 1]

        x = x.reshape([20, 20])
        y = y.reshape([20, 20])
        y_idx = np.zeros(len(y))

        ax.set_xlabel('q_r')
        ax.set_ylabel('q_d')
        cs = ax.contourf(x, y, values)
        ax.scatter(x, y, c=values, cmap=matplotlib.colors.ListedColormap(colours))
        ax.set_title("Levelset Constraint 2")

        # fig.colorbar(cs, ax=ax, shrink=0.9)
        ax.set_ylim([-0.01, 2.01])
        plt.savefig('Safeset' + data_type+'.png', dpi=300)