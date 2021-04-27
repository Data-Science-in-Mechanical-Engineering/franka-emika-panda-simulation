import numpy as np
import gym
from panda_env import PandaEnv
import mujoco_py



def _Mx(M,J,threshold=1e-3):

    M_inv=np.linalg.inv(M)
    Mx_inv=np.dot(J,np.dot(M_inv,J.T))
    Mx=np.linalg.pinv(Mx_inv,rcond=1e-4)
    return Mx,Mx_inv


env=PandaEnv()
obs=env.reset()
joints=9
_MNN_vector = np.zeros(joints ** 2)
Kp=np.eye(3)*200 #Check out impedance controller: [100,800,1000]
Kd=np.sqrt(Kp)*2
controller_class=["impedance","gravity_compensation", "inverse_dynamics"]
controller = controller_class[2]
#Kp=1
for i in range(2000):
    action=env.action_space.sample()
    mujoco_py.cymj._mj_fullM(env.model,_MNN_vector,env.sim.data.qM)
    M=_MNN_vector
    M = M.reshape((joints, joints))
    bias = -1 * env.sim.data.qfrc_bias

    id=env.sim.model.site_name2id("panda:grip")
    J=env.sim.data.site_jacp[id,:]
    J=J.reshape(3,-1)
    u = -bias
    Mx,Mx_inv=_Mx(M,J)

    wM_des = np.dot(Kp, (obs["desired_goal"] - obs["achieved_goal"]))-np.dot(Kd,obs["observation"][3:])

    if controller==controller_class[0]: #Impedance control
        u+=np.dot(J.T,wM_des)

    elif controller==controller_class[2]: #Inverse dynamics control
        u += np.dot(J.T,np.dot(Mx,wM_des))



    #u+=-10*env.sim.data.qvel
    torque=u
    torque=np.clip(torque,env.action_space.low,env.action_space.high)
    obs,reward,done,info=env.step(torque)
    env.render()
    u_prev=u

print((obs["desired_goal"]-obs["achieved_goal"]))