import numpy as np
import gym
import Panda_Env #Library defined for the panda environment
import mujoco_py
import random

import os
random.seed(0)
np.random.seed(0)
import scipy
from osc_controller import inverse_dynamics_control
from casadi import *
def _Mx(M,J):
    M_inv=np.linalg.inv(M)
    Mx_inv=np.dot(J,np.dot(M_inv,J.T))
    Mx=np.linalg.pinv(Mx_inv,rcond=1e-4)
    return Mx,Mx_inv


os.chdir(os.path.dirname(os.path.realpath(__file__)))
env=gym.make("PandaEnv-v0")

env.seed(0)
obs=env.reset()
#()
joints=9
_MNN_vector = np.zeros(joints ** 2)

approximate=False

ID=inverse_dynamics_control(env=env,njoints=joints,target=obs["desired_goal"])
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


N=20
Q1=np.linspace(0,6,N)
Q2=np.linspace(0,2,N)

q_vals=np.asarray(np.meshgrid(Q1,Q2)).T.reshape(-1,2)
data=np.zeros([N**2,5])
data[:,:2]=q_vals
num_steps=3000
for j in range(N**2):
    obs = env.reset()
    init_dist = np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])

    Q_pos=np.power(10,data[j,0])
    Q_vel=np.sqrt(Q_pos)*data[j,1]
    Q=np.diag([Q_pos, Q_pos,Q_pos,Q_vel,Q_vel,Q_vel])
    R=np.eye(3)/100

    dist_factor = np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"]) / init_dist
    dist_con = (1 + np.exp(-dist_factor)) / 2
    vel_con=(1+np.exp(-np.linalg.norm(obs["observation"][3:])))/2
    #data[j,2]=np.linalg.norm(obs["desired_goal"]-obs["achieved_goal"])/init_dist
    data[j,2]=vel_con*dist_con
    P=np.matrix(scipy.linalg.solve_continuous_are(A,B,Q,R))
    K = scipy.linalg.inv(R)*(B.T*P)

    #Kp[2,2]=Kp[2,2]*1.5

    K=np.asarray(K)
    Kp=K[:,:3]
    Kd=K[:,3:]



    weighted=True #Forces some torques to be 0
    zero_vel=False



    #model_type = 'continuous' # either 'discrete' or 'continuous'
    #model = do_mpc.model.Model(model_type)
    #x=model.set_variable(var_type='_x', var_name='x', shape=(2*joints,1))
    #u=model.set_variable(var_type='_u', var_name='u', shape=(joints,1))
    #dphi_next = vertcat(ID.Dynamics(x,u))
    for i in range(num_steps):

        bias=ID.g()
        M=ID.M()

        #data[j, 2] = np.maximum(data[j,2],np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])/init_dist)

        id=env.sim.model.site_name2id("panda:grip")
        #J=env.sim.data.site_jacp[id,:]
        #J=J.reshape(3,-1)
        J=ID.Jp(id)
        u = -bias
        Mx,Mx_inv=_Mx(M,J)

        wM_des = np.dot(Kp, (obs["desired_goal"] - obs["achieved_goal"]))-np.dot(Kd,obs["observation"][3:])

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
            T1[4] = 1
            T1[6:] = 1
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
        data[j, 4]+=reward
        dist_factor=np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])/init_dist
        dist_con = (1 + np.exp(-dist_factor)) / 2
        vel_con = (1 + np.exp(-np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2
        # data[j,2]=np.linalg.norm(obs["desired_goal"]-obs["achieved_goal"])/init_dist
        data[j, 2] = np.maximum(vel_con * dist_con, data[j, 2])
        #print(np.hstack((env.sim.data.qpos[4], env.sim.data.qpos[6:])))

        data[j,3]=np.maximum(np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])-init_dist,data[j,3])

    #if info["is_success"]:
    #    print("reached target")
    #    env.reset()
    data[j,4]*=1/num_steps
    #data[j, 3] = np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])

    print((obs["desired_goal"]-obs["achieved_goal"]))


np.savetxt('data_arm_full.csv', data, delimiter=',')