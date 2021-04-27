import numpy as np
import gym
from panda_env import PandaEnv
import mujoco_py



def _Mx(M,J,threshold=1e-1):

    M_inv=np.linalg.inv(M)
    Mx_inv=np.dot(J,np.dot(M_inv,J.T))
    Mx=np.linalg.pinv(Mx_inv,rcond=threshold*0.1)
    return  Mx,Mx_inv


env=PandaEnv()
obs=env.reset()
joints=9
_MNN_vector = np.zeros(joints ** 2)
Kp=np.eye(3)*0.5
Kd=np.sqrt(Kp)*2

for i in range(1000):
    action=env.action_space.sample()
    mujoco_py.cymj._mj_fullM(env.model,_MNN_vector,env.sim.data.qM)
    M=_MNN_vector
    M = M.reshape((joints, joints))
    bias = -1 * env.sim.data.qfrc_bias
    u = np.zeros(joints)
    id=env.sim.model.site_name2id("panda:grip")
    J=env.sim.data.site_jacp[id,:]
    J=J.reshape(3,-1)

    Mx,Mx_inv=_Mx(M,J)

    wM_des = np.dot(Kp, (obs["desired_goal"] - obs["achieved_goal"]))-np.dot(Kd,obs["observation"][3:])
    #Jr=env.sim.data.site_jacr[id,:]
    #Jr=Jr.reshape(3,-1)
    #u = np.dot(Jr.T,np.dot(Mx,wM_des))*0.5
    # Gravity compensation through torque
    u+=-bias
    #u+=-10*env.sim.data.qvel
    torque=u
    torque=np.clip(torque,env.action_space.low,env.action_space.high)
    obs,reward,done,info=env.step(torque)
    env.render()
    u_prev=u

print((obs["desired_goal"]-obs["achieved_goal"]))