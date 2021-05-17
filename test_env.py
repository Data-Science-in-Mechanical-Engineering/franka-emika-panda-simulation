import numpy as np
import gym
import Panda_Env #Library defined for the panda environment
import mujoco_py
import random


random.seed(0)
np.random.seed(0)
import scipy
from osc_controller import inverse_dynamics_control

def _Mx(M,J):
    M_inv=np.linalg.inv(M)
    Mx_inv=np.dot(J,np.dot(M_inv,J.T))
    Mx=np.linalg.pinv(Mx_inv,rcond=1e-4)
    return Mx,Mx_inv



env=gym.make("PandaEnv-v0")

env.seed(0)
obs=env.reset()
#()
joints=9
_MNN_vector = np.zeros(joints ** 2)


ID=inverse_dynamics_control(env=env,njoints=joints,target=obs["desired_goal"])
q_des=ID.find_qdes()
id=env.sim.model.site_name2id("panda:grip")
J=ID.Jp(id)
A=np.zeros([6,6])
#A=np.zeros([18,18])
A[:3,3:]=np.eye(3)
B=np.zeros([6,3])

#Ts=1/1000
region=2
#Ad=np.eye(joints*2)+A*Ts

#qr_region_1=2.526
#qd_region_1=1.263
qr_region_1=6
qd_region_1=1.5



B[3:,:]=np.eye(3)

qr_region_2=3.15784
qd_region_2=0

if region==1:
    Q_pos=np.power(10,qr_region_1)
    Q_vel=np.sqrt(Q_pos)*qd_region_1

else:
    Q_pos = np.power(10, qr_region_2)
    Q_vel = np.sqrt(Q_pos) * qd_region_2


Q=np.diag([Q_pos, Q_pos,Q_pos,Q_vel,Q_vel,Q_vel])
R=np.eye(3)/100
approximate=True


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
controller = controller_class[2]

weighted=True
zero_vel=False




for i in range(3000):
    action=env.action_space.sample()
    bias=ID.g()
    M=ID.M()


    id=env.sim.model.site_name2id("panda:grip")

    J=ID.Jp(id)
    u = -bias
    Mx,Mx_inv=_Mx(M,J)

    wM_des = np.dot(Kp, (obs["desired_goal"] - obs["achieved_goal"]))-np.dot(Kd,obs["observation"][3:])

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
        T1 = np.zeros(9)
        T1[4] = 1
        T1[6:] = 1
        T = np.diag(T1)
        N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
        N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))
        torque=np.dot(N_bar,u)
    else:
        torque=u

    obs,reward,done,info=env.step(torque)
    env.render()





print((obs["desired_goal"]-obs["achieved_goal"]))

