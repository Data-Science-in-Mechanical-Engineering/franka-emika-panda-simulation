import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from gosafeopt import SafeOptSwarm, GoSafeOptPractical
import gym
import pandaenv #Library defined for the panda environment
import mujoco_py
import scipy
from pandaenv.utils import inverse_dynamics_control
import GPy
import random
import time
import logging
import pandas as pd
from classireg.acquisitions.expected_improvement_with_constraints import ExpectedImprovementWithConstraints
from classireg.utils.parsing import convert_lists2arrays, save_data, get_logger
import torch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models import ModelListGP
class System(object):

    def __init__(self,position_bound,velocity_bound,rollout_limit=0,upper_eigenvalue=0):
        self.env = gym.make("PandaEnvPath-v0")
        self.Q=np.eye(6)
        self.R=np.eye(3)/100
        self.env.seed(0)
        self.obs = self.env.reset()
        self.A = np.zeros([6, 6])
        # A=np.zeros([18,18])
        self.A[:3, 3:] = np.eye(3)
        self.B = np.zeros([6, 3])
        self.B[3:, :] = np.eye(3)
        self.T=2000
        self.ID = inverse_dynamics_control(env=self.env, njoints=9, target=self.env.goal)
        self.id = self.env.sim.model.site_name2id("panda:grip")
        self.rollout_limit=rollout_limit
        self.at_boundary=False
        self.Fail=False
        self.approx=True
        self.position_bound=position_bound
        self.velocity_bound=velocity_bound
        self.upper_eigenvalue=upper_eigenvalue
        self.rho_max=1/100
        self.rho_min=1/500
        self.kappa_max=1
        self.kappa_min=0
        self.boundary_frequency=1
        T1 = np.zeros(9)
        T1[4] = 1
        T1[6:] = 1
        T = np.diag(T1)
        N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
        self.N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))

    def simulate(self,params=None,opt=None,update=False):
        x0=None
        if params is not None:
            if update:
                param_a=self.set_params(params)
                self.Q = np.diag(param_a)
            else:
                self.Q=np.diag(params)

            self.R=np.eye(3)/100*np.power(10,3*params[2]) #param is between -1 and 1
            self.env.rho_action=params[3]*(self.rho_max-self.rho_min)+self.rho_min
            if opt is not None:
                if opt.criterion in ["S2"]:
                    x0=params[opt.state_idx]
                    x0[3:]=np.zeros(3)


        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
        K = scipy.linalg.inv(self.R) * (self.B.T * P)


        K = np.asarray(K)
        Kp = K[:, :3]
        Kd = K[:, 3:]
        Objective=0
        self.reset(x0)
        state = []
        constraint2 = 0
        if x0 is not None:
            rho = np.asarray([self.env.rho]).reshape(1, -1)
            x=np.hstack([params[:4].reshape(1,-1),x0.reshape(1,-1),rho])
            state.append(x)

        else:
            obs = self.obs["observation"].copy()
            obs[:3] /= self.position_bound
            obs[3:] /= self.velocity_bound
            rho=np.asarray([self.env.rho]).reshape(1,-1)
            x = np.hstack([params[:4].reshape(1, -1), obs.reshape(1, -1),rho])
            state.append(x)




        #init_dist = self.init_dist
        constraint2 = np.zeros(self.rollout_limit + 1)
        Objective = np.zeros(self.rollout_limit + 1)
        for i in range(self.T):
            if opt is not None and not self.at_boundary:
                if i % self.boundary_frequency == 0:
                    obs=self.obs["observation"].copy()
                    obs[:3]/=self.position_bound
                    obs[3:]/=self.velocity_bound
                    rho = np.asarray([self.env.rho]).reshape(1, -1)
                    obs = np.hstack([obs.reshape(1, -1), rho])
                    self.at_boundary, self.Fail, params = opt.check_rollout(state=obs, action=params)

                if self.Fail:
                    print("FAILED                  ",i,end=" ")
                    return 0, 0,0,state
                elif self.at_boundary:
                    params = params.squeeze()
                    print(" Changed action to",i,params,'constraint',constraint2[0], end="")
                    param_a = self.set_params(params.squeeze())
                    self.Q = np.diag(param_a)
                    self.R=np.eye(3) / 100 * np.power(10, 3 * params[2])
                    self.env.rho_action=params[3]*(self.rho_max-self.rho_min)+self.rho_min
                    P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
                    K = scipy.linalg.inv(self.R) * (self.B.T * P)

                    K = np.asarray(K)

                    Kp = K[:, :3]
                    Kd = K[:, 3:]


            if i>0 and i < self.rollout_limit:
                obs=self.obs["observation"].copy()
                obs[:3] /= self.position_bound
                obs[3:] /= self.velocity_bound
                rho = np.asarray([self.env.rho]).reshape(1, -1)
                x=np.hstack([params[:4].reshape(1,-1),obs.reshape(1,-1),rho])
                state.append(x)
                constraint2[i] = 0
                Objective[i] = 0
            bias = self.ID.g()


            J = self.ID.Jp(self.id)

            wM_des = -np.dot(Kp, (self.obs["observation"][:3])) - np.dot(Kd, self.obs["observation"][3:])
            u=-bias
            u += np.dot(J.T, wM_des)
            u = np.dot(self.N_bar, u)
            u= np.clip(u, self.env.action_space.low, self.env.action_space.high)
            self.obs, reward, done, info = self.env.step(u)
            Objective += reward
            constraint2 = np.maximum(constraint2, np.linalg.norm(self.obs["observation"][:3])*np.ones(self.rollout_limit+1))

            #constraint2 = np.maximum(np.max(np.abs(self.obs["observation"][3:])), constraint2)





        return Objective/self.T,constraint2,state


    def reset(self,x0=None):
        self.obs = self.env.reset()
        #self.init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
        self.Fail=False
        self.at_boundary=False

        if x0 is not None:
            x0*=self.position_bound
            self.env.goal=self.obs["observation"][:3]-x0[:3]




    def set_params(self, params):
        q1 = np.repeat(np.power(10, 6*params[0]),3) #param is between -1 and 1
        q2 = np.sqrt(q1)*params[1]*(self.kappa_max-self.kappa_min)+self.kappa_min
        updated_params = np.hstack((q1.squeeze(), q2.squeeze()))
        return updated_params

class ExactGPModel(gpytorch.models.ExactGP,GPyTorchModel):
    '''
    Define ExactGP from gpytorch module
    '''
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.dim=train_x.shape[1]
        lengthscale = torch.tensor([ [0.4/6,0.2,0.4/3,0.2]])
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.dim,lengthscale=lengthscale))
        self.eval()
        self.train_ys=train_y
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class optimization(object):
    def __init__(self):
        self.Nrestarts=4
        self.algo_name= 'LN_BOBYQA'
        self.disp_info_scipy_opti= False # Display info about the progress of the scipy optimizer
class acqui_options(object):
    def __init__(self):
        self.optimization=optimization()
        self.prob_satisfaction= 0.98


class Eic(object):
    '''
    EIC optimizer
    '''
    def __init__(self, error_bound=0.25, lengthscale=0.4, ARD=True):
        self.error_bound=error_bound
        q =4/6
        r = -1
        kappa=0.1
        rho_action=0
        self.params = np.asarray([q,kappa,r,rho_action])
        self.failures=0
        self.failure_overshoot = 0
        self.rollout_limit = 500
        self.mean_reward = -0.3
        self.std_reward = 0.1

        self.position_bound = 0.3
        self.velocity_bound = 5
        self.sys = System(rollout_limit=self.rollout_limit,position_bound=self.position_bound,velocity_bound=self.velocity_bound)
        f, g1, state = self.sys.simulate(self.params,update=True)
        f=f[0]
        g1=g1[0]
        g1-=self.error_bound
        g1=g1/self.error_bound
        f -= self.mean_reward
        f /= self.std_reward
        f*=-1
        x = self.params.reshape(1, -1)
        x = torch.from_numpy(x).float()
        x = torch.reshape(x, (1, -1))
        y_f = torch.tensor([f]).float()
        y_g = torch.tensor([g1]).float()
        self.gp_obj = ExactGPModel(train_x=x.clone(), train_y=y_f.clone())
        self.gp_con = ExactGPModel(train_x=x.clone(), train_y=y_g.clone())
        self.constraints = {1: (None, 0)}
        model_list = ModelListGP(self.gp_obj, self.gp_con)
        self.acqui_options = acqui_options()
        self.eic = ExpectedImprovementWithConstraints(model_list=model_list, constraints=self.constraints,
                                                      options=self.acqui_options)
        self.time_recorded = []
        self.simulate_data()

    def simulate_data(self):
        p = [5 / 6, 1]
        d = [-0.9, -2 / 3]
        kappa = 0.5
        rho_action = 1
        for i in range(2):
            self.params = np.asarray([p[i], kappa, d[i], rho_action])
            f, g1, state = self.sys.simulate(self.params, update=True)
            f=f[0]
            g1=g1[0]
            g1 -= self.error_bound
            g1 = g1 / self.error_bound
            f -= self.mean_reward
            f /= self.std_reward
            f*=-1
            print(f, g1)
            y = np.array([[f], [g1]])
            y = y.squeeze()
            self.add_data(self.params.reshape(1, -1), y)

    def optimize(self):
        '''
        Run 1 optimization step
        '''
        start_time = time.time()
        param, val = self.eic.get_next_point()
        param = param.numpy().squeeze()
        self.time_recorded.append(time.time() - start_time)
        print(param, end="")
        f, g1, state = self.sys.simulate(param, update=True)
        # print(f, g1, g2,self.opt.criterion)
        f = f[0]
        g1 = g1[0]
        g1 -= self.error_bound
        g1 = g1 / self.error_bound
        f -= self.mean_reward
        f /= self.std_reward
        f *= -1
        y = np.array([f, g1])
        y = y.squeeze()
        constraint_satisified = g1 <= 0
        if not constraint_satisified:
            self.failure_overshoot += (g1 * self.error_bound) + self.error_bound
        self.failures += constraint_satisified
        print(f, g1, constraint_satisified)
        self.add_data(param.reshape(1, -1), y, constraint_satisified)

    def add_data(self,x,y,constraint_satisified=True):
        x=torch.from_numpy(x).float()
        y_f = torch.tensor([y[0]]).float()
        y_g = torch.tensor([y[1]]).float()
        xx=torch.reshape(torch.stack(list(self.gp_con.train_inputs), dim=0),(-1,x.shape[1]))
        train_x = torch.cat([xx, x])
        train_yl_cons = torch.cat([self.gp_con.train_targets, y_g], dim=0)
        train_yl_f=torch.cat([self.gp_obj.train_targets,y_f],dim=0)
        self.gp_obj.set_train_data(inputs=train_x, targets=train_yl_f, strict=False)
        self.gp_obj.eval()
        self.gp_con.set_train_data(inputs=train_x, targets=train_yl_cons, strict=False)
        self.gp_con.eval()
        #self.gp_obj.train_ys=train_yl_f
        if constraint_satisified:
            self.gp_con.train_ys=train_yl_cons
            self.gp_obj.train_ys=train_yl_f

        #self.gp_obj=GPmodel(dim=train_x_cons_new.shape[1], train_X=train_x_cons_new.clone(), train_Y=train_yl_f.clone(), options=self.cfg.gpmodel,nu=1.5)
        #self.gp_con=GPCRmodel(dim=train_x_cons_new.shape[1], train_x=train_x_cons_new.clone(), train_yl=train_yl_cons_new.clone(), options=self.cfg.gpcr_model,nu=1.5)
        model_list = ModelListGP(self.gp_obj, self.gp_con)
        self.eic = ExpectedImprovementWithConstraints(model_list=model_list, constraints=self.constraints,
                                                      options=self.acqui_options)










#opt=SafeOpt_Optimizer()
method="eic"
#method="GoSafe"
iterations=201
runs=10
plot=False

Reward_data = np.zeros([41, runs])
Overshoot_summary = np.zeros([2, runs])
for r in range(runs):
    j=0
    opt = Eic()
    random.seed(r)
    np.random.seed(r)
    opt.sys.env.seed(r)
    torch.manual_seed(r)
    for i in range(iterations):
        if i%5==0:
            ind = torch.argmin(opt.gp_obj.train_ys)
            maximum = opt.gp_obj.train_inputs[0][ind.item(), :].numpy()
            f, g1, dummy = opt.sys.simulate(maximum, update=True)
            f=f[0]
            f -= opt.mean_reward
            f /= opt.std_reward
            Reward_data[j, r] = f
            j+=1
        opt.optimize()
        print(i)
    print(opt.failures / iterations,r)

    Overshoot_summary[0, r] = opt.failures / iterations
    failure=np.maximum(1e-3,iterations-opt.failures)
    Overshoot_summary[1, r] = opt.failure_overshoot / (failure)
    ind = torch.argmin(opt.gp_obj.train_ys)
    maximum = opt.gp_obj.train_inputs[0][ind.item(), :].numpy()
    f, g1, dummy = opt.sys.simulate(maximum, update=True)
    f = f[0]
    f -= opt.mean_reward
    f /= opt.std_reward
    print(maximum, f)
np.savetxt('eic_Overshoot.csv', Overshoot_summary, delimiter=',')
np.savetxt('eic_Reward.csv', Reward_data, delimiter=',')





print(opt.failures/iterations)
#max,f=opt.opt.get_maximum()
#max=[2.00179108e+00,4.13625539e+00, 3.34599393e+00, 7.41304209e-01,2.81500345e-01, 3.13137132e-03]
time_recorder=np.asarray(opt.time_recorded)
print("Time:",time_recorder.mean(),time_recorder.std())
print("maximum",max)

#f,g1,g2,state=opt.sys.simulate(max,update=True,render=True)





