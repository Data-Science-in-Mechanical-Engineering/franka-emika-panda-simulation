import numpy as np
import gym
import Panda_Env #Library defined for the panda environment
import mujoco_py
import random
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from scipy.interpolate import interp1d
random.seed(0)
np.random.seed(0)
import scipy
from osc_controller import inverse_dynamics_control

def _Mx(M,J):
    M_inv=np.linalg.inv(M)
    Mx_inv=np.dot(J,np.dot(M_inv,J.T))
    Mx=np.linalg.pinv(Mx_inv,rcond=1e-4)
    return Mx,Mx_inv



env=gym.make("PandaEnvPath-v0")

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

ID.penalty=1e-2
q_des=ID.find_qdes()
A=np.zeros([6,6])
#A=np.zeros([18,18])
A[:3,3:]=np.eye(3)
B=np.zeros([6,3])
B[3:,:]=np.eye(3)

#Ts=1/1000
region=2 #region=2 gosafe maximum
#Ad=np.eye(joints*2)+A*Ts

#qr_region_1=2.526
#qd_region_1=1.263



#q_2=6.15784
#r_2=9.47
q_2=6*1
r_2=np.power(10.,3*0.44357401)




#q_1 = 6*0.98873
#r_1 = np.power(10.,3*0.01695)
q_1=6
r_1 = np.power(10.,3*-1)

kappa=0.0

if region==1:
    Q_pos=np.power(10,q_1)
    Q_vel=np.sqrt(Q_pos)*kappa
    R = np.eye(3) / 100*r_1
    rho_a = 0.98495133* (1 / 100 - 1 / 500) + 1 / 500

else:
    Q_pos = np.power(10., q_2)
    Q_vel = np.sqrt(Q_pos) * kappa
    R = np.eye(3) / 100 * r_2
    rho_a = 0.09989707 * (1 / 100 - 1 / 500) + 1 / 500

Q=np.diag([Q_pos, Q_pos,Q_pos,Q_vel,Q_vel,Q_vel])


P=np.matrix(scipy.linalg.solve_continuous_are(A,B,Q,R))
K = scipy.linalg.inv(R)*(B.T*P)

#Kp[2,2]=Kp[2,2]*1.5

K=np.asarray(K)

#Kp=np.eye(3)*300 #Check out impedance controller: [100,800,1000] , inverse dynamics controller [200 500 1000]
#Kp[0,0]=Kp[0,0]
#Kp[1,1]=Kp[1,1]
#Kp[2,2]=Kp[2,2]
#Kd=np.sqrt(Kp)*2

Kp=K[:,:3]
Kd=K[:,3:]
controller_class=["impedance","gravity_compensation", "inverse_dynamics_task_space","inverse_dynamics_joint_space"]
controller = controller_class[0]

weighted=True


init_dist = np.linalg.norm(env.goal - obs["achieved_goal"])
dist_factor = np.linalg.norm(env.goal - obs["achieved_goal"]) / init_dist
dist_con = (1 + np.exp(-dist_factor)) / 2
vel_con = (1 + np.exp(-0.5*np.tanh(np.linalg.norm(obs["observation"][3:])))) / 2

#print(obs["observation"][3:])
constraint_1=dist_con*vel_con
constraint_2=0
Total_reward=0
contraint_3=0
constraint_4=0
constraint_5=0

num_steps=5000

approximate=False


error=0


env.rho_action=rho_a
Total_reward=0
#delta_z=lambda x: np.sin(x/delta_x*np.pi)
reference=np.zeros([6,num_steps])
states=np.zeros([6,num_steps])
for i in range(num_steps):
    states[:,i]=np.hstack((obs["achieved_goal"],obs["velocity_EE"]))
    reference[:3,i]=obs["desired_goal"]
    reference[3:,i]=obs["velocity_goal"]
    # if i<500:
    #     obs["desired_goal"]=obs["achieved_goal"].copy()
    #     obs["desired_goal"][2]=env.goal[2]
    #
    # else:
    #     obs["desired_goal"][1]=f(obs["desired_goal"][0])

    #x_dist = env.goal[0] - obs["achieved_goal"][0]



    bias = ID.g()
    M = ID.M()

    # data[j, 2] = np.maximum(data[j,2],np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])/init_dist)

    id = env.sim.model.site_name2id("panda:grip")
    # J=env.sim.data.site_jacp[id,:]
    # J=J.reshape(3,-1)
    J = ID.Jp(id)
    u = -bias
    Mx, Mx_inv = ID.Mx(M, J)

    wM_des = -np.dot(Kp, (obs["observation"][:3])) - np.dot(Kd, obs["observation"][3:])
    error=np.maximum(error,np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"]))
    #error = np.maximum(error, np.linalg.norm(obs["observation"][:3]))
    #ddq_des=200*(q_des-env.sim.data.qpos)-2*np.sqrt(200)*env.sim.data.qvel
    if controller == controller_class[0]:  # Impedance control
        u += np.dot(J.T, wM_des)
    #u+=np.dot(M,ddq_des)
    elif controller == controller_class[2]:  # Inverse dynamics control
        if approximate:
            diag_M = Mx.diagonal()
            approx_M = np.diag(diag_M)
            u += np.dot(J.T, np.dot(approx_M, wM_des))
        else:
            u += np.dot(J.T, np.dot(Mx, wM_des))
    # elif controller==controller_class[3]:
    #    x=np.hstack((env.sim.data.qpos,env.sim.data.qvel))
    #    x[:joints]=x[:joints]-q_des

    #    u+=np.dot(M,np.dot(-K,x))

    if weighted:
        T1 = np.zeros(9)
        T1[4] =1
        T1[6:] = 1
        T = np.diag(T1)
        N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
        N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))
        torque = np.dot(N_bar, u)
    else:
        torque = u


    torque=np.clip(torque,env.action_space.low,env.action_space.high)
    #noise=np.maximum(np.random.normal(size=9)*torque,0.1*np.ones(9))

    obs, reward, done, info = env.step(torque)
    Total_reward+=reward
    env.render()


print(error)
print(Total_reward/2000)
#print((obs["achieved_goal"]-env.goal))
#print((obs["desired_goal"]-obs["achieved_goal"]),constraint_1,constraint_2,contraint_3,Total_reward)

fig, axs = plt.subplots(6, sharex=True,figsize=(12, 9.6))
fig.suptitle("State Evolution GoSafe")
ylabel=['$r_{e,x}$','$r_{e,y}$','$r_{e,z}$','$v_{e,x}$','$v_{e,y}$','$v_{e,z}$']
t=np.linspace(0,num_steps-1,num_steps)
for i in range(6):
    axs[i].plot(t,states[i,:],label="x" if i == 0 else "")
    axs[i].plot(t,reference[i,:],label="r" if i == 0 else "")
    #axs[i].set_xlabel('t')
    axs[i].set_ylabel(ylabel[i])

fig.legend(loc='center right')
plt.xlabel('$t$')
fig.show()
fig.align_ylabels(axs)
fig.savefig("GoSafeStates_path.png",dpi=300)