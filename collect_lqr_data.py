import numpy as np
import gym
import Panda_Env #Library defined for the panda environment
import mujoco_py
import random

import os
#random.seed(0)
#np.random.seed(0)
import scipy
from osc_controller import inverse_dynamics_control



os.chdir(os.path.dirname(os.path.realpath(__file__)))
env=gym.make("PandaEnv-v0")

#env.seed(0)
obs=env.reset()
#()
joints=9


approximate=True
weighted=True #Forces some torques to be 0
ID=inverse_dynamics_control(env=env,njoints=joints,target=env.goal)
id=env.sim.model.site_name2id("panda:grip")
J=ID.Jp(id)
A=np.zeros([6,6])
#A=np.zeros([18,18])
A[:3,3:]=np.eye(3)
B=np.zeros([6,3])
B[3:,:]=np.eye(3)
#Ts=1/1000

#Ad=np.eye(joints*2)+A*Ts
controller_class = ["impedance", "gravity_compensation", "inverse_dynamics_task_space", "inverse_dynamics_joint_space"]
controller = controller_class[2]


Nsteps=25
Q1=np.linspace(-6,6,Nsteps)
Q2=np.linspace(-3,3,Nsteps)


q_vals=np.asarray(np.meshgrid(Q1,Q2)).T.reshape(-1,2)



num_steps=2000
Full_data=np.zeros([Nsteps ** 2, 10])
runs=1
Full_data[:,:2]=q_vals
for r in range(runs):
    env.seed(r)
    random.seed(r)
    np.random.seed(r)
    data = np.zeros(Full_data.shape)
    data[:, :2] = q_vals
    for j in range(int(Nsteps**2)):
        obs = env.reset()
        init_dist = np.linalg.norm(env.goal - obs["achieved_goal"])

        Q_pos=np.power(10,data[j,0])
        Q_vel=np.sqrt(Q_pos)*0.1

        Q=np.diag([Q_pos, Q_pos,Q_pos,Q_vel,Q_vel,Q_vel])
        R=np.eye(3)/100*np.power(10,data[j,1])

        dist_factor = np.linalg.norm(env.goal - obs["achieved_goal"]) / init_dist
        dist_con = (1 + np.exp(-dist_factor)) / 2
        vel_con = (1 + np.exp(-0.5*np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2
        #data[j,2]=np.linalg.norm(obs["desired_goal"]-obs["achieved_goal"])/init_dist
        data[j,2]=vel_con*dist_con
        vel_con = (1 + np.exp(-0.75*np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2
        data[j,3]=vel_con*dist_con
        vel_con = (1 + np.exp(-np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2
        data[j, 4] = vel_con * dist_con
        P=np.matrix(scipy.linalg.solve_continuous_are(A,B,Q,R))
        K = scipy.linalg.inv(R)*(B.T*P)

        K=np.asarray(K)
        Kp=K[:,:3]
        Kd=K[:,3:]
        eigen_value = np.linalg.eig(A - np.dot(B, K))
        eigen_value = np.max(np.asarray(eigen_value[0]).real)
        data[j,9]=eigen_value
        for i in range(num_steps):

            bias=ID.g()
            M=ID.M()

            id=env.sim.model.site_name2id("panda:grip")

            J=ID.Jp(id)
            u = -bias
            Mx,Mx_inv=ID.Mx(M,J)

            wM_des = np.dot(Kp, (obs["desired_goal"] - obs["achieved_goal"]))-np.dot(Kd,obs["observation"][3:]-np.ones(3)*1/env.Tmax*(i<env.Tmax))

            if controller==controller_class[0]: #Impedance control
                u+=np.dot(J.T,wM_des)

            elif controller==controller_class[2]: #Inverse dynamics control
                if approximate:
                    diag_M = Mx.diagonal()
                    approx_M = np.diag(diag_M)
                    u += np.dot(J.T, np.dot(approx_M, wM_des))
                else:
                    u += np.dot(J.T, np.dot(Mx, wM_des))
            #elif controller==controller_class[3]:
            #    x=np.hstack((env.sim.data.qpos,env.sim.data.qvel))
            #    x[:joints]=x[:joints]-q_des

            #    u+=np.dot(M,np.dot(-K,x))

            if weighted:
                T1 = np.zeros(9)
                T1[4:] = 1
                #T1[6:] = 1
                T = np.diag(T1)
                N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
                N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))
                torque=np.dot(N_bar,u)
            else:
                torque=u

            #print(torque)
            #torque=np.clip(torque,env.action_space.low,env.action_space.high)


            obs,reward,done,info=env.step(torque)
            #env.render()

            dist_factor=np.linalg.norm(env.goal - obs["achieved_goal"])/init_dist
            dist_con = (1 + np.exp(-dist_factor)) / 2
            vel_con = (1 + np.exp(-0.5*np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2
            # data[j,2]=np.linalg.norm(obs["desired_goal"]-obs["achieved_goal"])/init_dist
            data[j, 2] = np.maximum(vel_con * dist_con, data[j, 2])
            #print(np.hstack((env.sim.data.qpos[4], env.sim.data.qpos[6:])))
            vel_con = (1 + np.exp(-0.75 * np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2
            data[j, 3] = np.maximum(vel_con * dist_con,data[j,3])
            vel_con = (1 + np.exp(-np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2
            data[j, 4] = np.maximum(vel_con * dist_con,data[j,4])
            data[j,5]=np.maximum(np.linalg.norm(env.goal - obs["achieved_goal"])-init_dist,data[j,5])
            data[j, 6] += reward
            input_bounds = (np.abs(torque) - env.action_space.high)/env.action_space.high
            input_bounds = np.max(input_bounds)


            data[j,7]=np.maximum(data[j,7],input_bounds)
            data[j,8]=np.maximum(np.max(np.abs(obs["observation"][3:])),data[j,8])
        #if info["is_success"]:
        #    print("reached target")
        #    env.reset()
        data[j,6]*=1/num_steps
        data[j, 5] = data[j, 5] / init_dist
        #data[j, 3] = np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])

        #print((obs["desired_goal"]-obs["achieved_goal"]),data[j,:])

    Full_data[:,2:]+=data[:,2:]

Full_data[:,2:]=Full_data[:,2:]/runs
np.savetxt('arm_cost_approximate_inputs.csv', Full_data, delimiter=',')