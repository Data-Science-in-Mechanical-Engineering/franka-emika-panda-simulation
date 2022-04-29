import numpy as np
import gym
import pandaenv #Library defined for the panda environment
import mujoco_py
import random
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
random.seed(0)
np.random.seed(0)
import scipy
from pandaenv.utils import inverse_dynamics_control

def _Mx(M,J):
    M_inv=np.linalg.inv(M)
    Mx_inv=np.dot(J,np.dot(M_inv,J.T))
    Mx=np.linalg.pinv(Mx_inv,rcond=1e-4)
    return Mx,Mx_inv

'''
Loads Panda environment and compares optimum found by safeopt (region1) and our method (region 2)
'''

env=gym.make("PandaEnvBasic-v0")

env.seed(0)
obs=env.reset()
env.render()
#()
joints=9
_MNN_vector = np.zeros(joints ** 2)

print(env.Tmax)
ID=inverse_dynamics_control(env=env,njoints=joints,target=env.goal)
id=env.sim.model.site_name2id("panda:grip")
J=ID.Jp(id)
A=np.zeros([6,6])
#A=np.zeros([18,18])
A[:3,3:]=np.eye(3)
B=np.zeros([6,3])
B[3:,:]=np.eye(3)

#Ts=1/1000
region=2 #region=2 gosafe maximum

q_2=6*1
r_2=np.power(10.,3*0.56)


q_1=6
r_1 = np.power(10.,3*0.14768434)

if region==1:
    Q_pos=np.power(10,q_1)
    Q_vel=np.sqrt(Q_pos)*0.1
    R = np.eye(3) / 100*r_1

else:
    Q_pos = np.power(10., q_2)
    Q_vel = np.sqrt(Q_pos) * 0.1
    R = np.eye(3) / 100 * r_2

Q=np.diag([Q_pos, Q_pos,Q_pos,Q_vel,Q_vel,Q_vel])

approximate=True


P=np.matrix(scipy.linalg.solve_continuous_are(A,B,Q,R))
K = scipy.linalg.inv(R)*(B.T*P)

K=np.asarray(K)


Kp=K[:,:3]
Kd=K[:,3:]

controller_class=["impedance","gravity_compensation", "inverse_dynamics_task_space","inverse_dynamics_joint_space"]
controller = controller_class[0]

# Sets torques 4, 6,7 equal to 0
weighted=True


init_dist = np.linalg.norm(env.goal - obs["achieved_goal"])
dist_factor = np.linalg.norm(env.goal - obs["achieved_goal"]) / init_dist
dist_con = (1 + np.exp(-dist_factor)) / 2
vel_con = (1 + np.exp(-0.5*np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2

print(obs["observation"][3:])
constraint_1=dist_con*vel_con
constraint_2=0
Total_reward=0
contraint_3=0
constraint_4=0
constraint_5=0

num_steps=2000
states=np.zeros([6,num_steps])
eigen_value=np.linalg.eig(A-np.dot(B,K))
eigen_value=np.max(np.asarray(eigen_value[0]).real)

reference=np.zeros([6,num_steps])
print(eigen_value)

for i in range(num_steps):
    states[:,i]=obs["observation"]
    reference[:,i]=np.hstack((obs["desired_goal"],np.ones(
        3) * 1 / env.Tmax * (i < env.Tmax)))
    bias = ID.g()
    M = ID.M()

    id = env.sim.model.site_name2id("panda:grip")
    J = ID.Jp(id)
    u = -bias
    Mx, Mx_inv = ID.Mx(M, J)

    wM_des = np.dot(Kp, (obs["desired_goal"] - obs["achieved_goal"])) - np.dot(Kd, obs["observation"][3:] - np.ones(
        3) * 1 / env.Tmax * (i < env.Tmax))

    if controller == controller_class[0]:  # Impedance control
        u += np.dot(J.T, wM_des)

    elif controller == controller_class[2]:  # Inverse dynamics control
        if approximate:
            diag_M = Mx.diagonal()
            approx_M = np.diag(diag_M)
            u += np.dot(J.T, np.dot(approx_M, wM_des))
        else:
            u += np.dot(J.T, np.dot(Mx, wM_des))


    if weighted:
        # Set torques 4,6,7,.. to 0
        T1 = np.zeros(9)
        T1[4] = 1
        T1[6:] = 1
        T = np.diag(T1)
        N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
        N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))
        torque = np.dot(N_bar, u)
    else:
        torque = u


    obs, reward, done, info = env.step(torque)
    env.render()
    Total_reward += reward
    dist_factor = np.linalg.norm(env.goal - obs["achieved_goal"]) / init_dist
    dist_con = (1 + np.exp(-dist_factor)) / 2
    vel_con = (1 + np.exp(-0.5*np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2

    constraint_1 = np.maximum(vel_con * dist_con, constraint_1)


    constraint_2 = np.maximum(np.linalg.norm(env.goal - obs["achieved_goal"]) - init_dist, constraint_2)
    input_bounds=(np.abs(torque) - env.action_space.high)/env.action_space.high
    input_bounds=np.max(input_bounds)
    contraint_3=np.maximum(contraint_3,input_bounds)

    constraint_4 = np.maximum(np.max(np.abs(obs["observation"][3:])),constraint_4)

Total_reward *= 1 / num_steps
constraint_2/=init_dist


# Plot state evoluation
fig, axs = plt.subplots(6, sharex=True,figsize=(12, 9.6))
fig.suptitle("State Evolution GoSafe")
ylabel=['$r_{e,x}$','$r_{e,y}$','$r_{e,z}$','$v_{e,x}$','$v_{e,y}$','$v_{e,z}$']
t=np.linspace(0,1999,2000)
for i in range(6):
    axs[i].plot(t,states[i,:],label="x" if i == 0 else "")
    axs[i].plot(t,reference[i,:],label="r" if i == 0 else "")
    #axs[i].set_xlabel('t')
    axs[i].set_ylabel(ylabel[i])
fig.legend(loc='center right')
plt.xlabel('$t$')
fig.show()
fig.align_ylabels(axs)
fig.savefig("GoSafeStates.png",dpi=300)
print(contraint_3,constraint_2,constraint_4)

