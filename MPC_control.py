import do_mpc
from casadi import *
import numpy as np
import gym
import Panda_Env #Library defined for the panda environment
import mujoco_py
import random
from osc_controller import inverse_dynamics_control
import sys, os
sys.stdout = open(os.devnull, 'w')
model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)
x=model.set_variable(var_type='_x', var_name='x', shape=(3,1))
u=model.set_variable(var_type='_u', var_name='u', shape=(3,1))
dx=model.set_variable(var_type='_x', var_name='dx', shape=(3,1))


model.set_rhs('x',dx)
model.set_rhs('dx',u)
model.setup()
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 20,
    't_step': 0.05,
    'n_robust': 1,
    'store_full_solution': False,
}
mpc.set_param(**setup_mpc)

mterm = (x[0]**2+ x[1]**2+ x[2]**2)*10000
lterm = (100*x[0]**2+ 100*x[1]**2+ 100*x[2]**2)*100 + (dx[0]**2+ dx[1]**2+ dx[2]**2)*1
mpc.set_objective(mterm=mterm, lterm=lterm)

kwargs={
'u':1e-4,
}
#mpc.set_rterm(**kwargs)
mpc.setup()
env=gym.make("PandaEnv-v0")

env.seed(0)
obs=env.reset()
ID=inverse_dynamics_control(env=env,njoints=9,target=obs["desired_goal"])
x0=np.hstack((obs["achieved_goal"]-obs["desired_goal"],obs["observation"][3:]))
mpc.x0 = x0
mpc.set_initial_guess()
for i in range(2000):
    action=env.action_space.sample()
    bias=ID.g()
    M=ID.M()

    x0 = np.hstack((obs["achieved_goal"]-obs["desired_goal"], obs["observation"][3:]))
    id=env.sim.model.site_name2id("panda:grip")

    J=ID.Jp(id)
    u = -bias
    Mx,Mx_inv=ID.Mx(M,J)

    wM_des=mpc.make_step(x0)
    wM_des=wM_des.squeeze()
    noise_matrix=np.random.uniform(0.0,1,Mx.shape)
    u += np.dot(J.T, np.dot(Mx*noise_matrix, wM_des))

    torque=u

    obs,reward,done,info=env.step(torque)
    env.render()