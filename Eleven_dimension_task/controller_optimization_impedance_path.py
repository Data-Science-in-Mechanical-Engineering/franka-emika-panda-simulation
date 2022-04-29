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
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
class System(object):

    def __init__(self,position_bound,velocity_bound,rollout_limit=0,upper_eigenvalue=0):
        self.env = gym.make("PandaEnvPath-v0")
        
        # Define controller parameters
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
        # Initialize inverse dynamics controller
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
        # Define weighting matrix to set torques 4,6,7,.. to 0
        T1 = np.zeros(9)
        T1[4] = 1
        T1[6:] = 1
        T = np.diag(T1)
        N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
        self.N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))

    def simulate(self,params=None,opt=None,update=False):
        '''
        Simulate system
        '''
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
        '''
        Reset system for next experiment
        '''
        self.obs = self.env.reset()
        #self.init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
        self.Fail=False
        self.at_boundary=False

        if x0 is not None:
            x0*=self.position_bound
            self.env.goal=self.obs["observation"][:3]-x0[:3]




    def set_params(self, params):
        '''
        Set Parameters for the system
        '''
        q1 = np.repeat(np.power(10, 6*params[0]),3) #param is between -1 and 1
        q2 = np.sqrt(q1)*params[1]*(self.kappa_max-self.kappa_min)+self.kappa_min
        updated_params = np.hstack((q1.squeeze(), q2.squeeze()))
        return updated_params




class SafeOpt_Optimizer(object):
    '''
    SafeOpt optimizer
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
        g1=-g1/self.error_bound
        f -= self.mean_reward
        f /= self.std_reward
        f = np.asarray([[f]])
        f = f.reshape(-1, 1)
        g1 = np.asarray([[g1]])
        g1 = g1.reshape(-1, 1)
        L = [lengthscale/6,0.2,lengthscale/3,0.2]
        x=self.params.reshape(1,-1)
        KERNEL_f = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        KERNEL_g = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        gp0 = GPy.models.GPRegression(x[0, :].reshape(1, -1), f, noise_var=0.1 ** 2, kernel=KERNEL_f)
        gp1 = GPy.models.GPRegression(x[0, :].reshape(1, -1), g1, noise_var=0.1 ** 2, kernel=KERNEL_g)

        bounds = [[1/3, 1], [0,1],[-1, 1],[0,1]]

        self.opt = SafeOptSwarm([gp0, gp1], fmin=[-np.inf, 0], bounds=bounds, beta=3)
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
            g1 = -g1 / self.error_bound
            f -= self.mean_reward
            f /= self.std_reward
            print(f, g1)
            y = np.array([[f], [g1]])
            y = y.squeeze()
            self.opt.add_new_data_point(self.params.reshape(1, -1), y)


    def optimize(self):
        '''
        Run 1 optimization step
        '''
        start_time = time.time()
        param = self.opt.optimize()
        self.time_recorded.append(time.time() - start_time)
        print(param, end="")
        f, g1,state = self.sys.simulate(param,update=True)
        #print(f, g1, g2,self.opt.criterion)
        f = f[0]
        g1 = g1[0]
        g1 -= self.error_bound
        g1 = -g1 / self.error_bound
        f -= self.mean_reward
        f /= self.std_reward


        y = np.array([[f], [g1]])
        y = y.squeeze()
        self.opt.add_new_data_point(param.reshape(1, -1), y)
        constraint_satisified = g1 >= 0
        if not constraint_satisified:
            self.failure_overshoot += -(g1* self.error_bound) + self.error_bound
        self.failures += constraint_satisified
        print(f, g1, constraint_satisified)


class GoSafeOpt_Optimizer(object):
    '''
    GoSafeOpt optimizer
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
        self.rollout_limit = 750
        self.mean_reward = -0.3
        self.std_reward = 0.1

        self.position_bound = 0.3
        self.velocity_bound = 5
        self.sys = System(rollout_limit=self.rollout_limit,position_bound=self.position_bound,velocity_bound=self.velocity_bound)
        f, g1, state = self.sys.simulate(self.params,update=True)

        g1-=self.error_bound
        g1=-g1/self.error_bound
        f -= self.mean_reward
        f /= self.std_reward
        print(f[0], g1[0])
        fscalar = f[0]
        g1scalar = g1[0]
        fscalar = np.asarray([[fscalar]])
        fscalar = fscalar.reshape(-1, 1)
        g1scalar = np.asarray([[g1scalar]])
        g1scalar = g1scalar.reshape(-1, 1)


        #g2 = np.asarray([[g2]])
        #g2 = g2.reshape(-1, 1)


        x0=state[0][:,4:]


        x=np.asarray(state).squeeze()

        L = [lengthscale / 6, 0.2, lengthscale / 3, 0.2, 0.5 / self.position_bound, 0.5 / self.position_bound,
             0.5 / self.position_bound,
             0.6, 0.6, 0.6, 10]

        L_states = np.asarray(L[4:])
        KERNEL_f = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        KERNEL_g = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        gp_full0 = GPy.models.GPRegression(x[0,:].reshape(1,-1), fscalar, noise_var=0.1 ** 2, kernel=KERNEL_f)
        gp_full1 = GPy.models.GPRegression(x[0,:].reshape(1,-1), g1scalar, noise_var=0.1 ** 2, kernel=KERNEL_g)

        # Try out larger bounds
        bounds = [[1 / 3, 1], [0, 1], [-1, 1], [0, 1]]

        a = x[:, :4]
        L = [lengthscale/6,0.2,lengthscale/3,0.2]
        KERNEL_f = GPy.kern.sde_RBF(input_dim=a.shape[1], lengthscale=L, ARD=ARD, variance=1)
        KERNEL_g = GPy.kern.sde_RBF(input_dim=a.shape[1], lengthscale=L, ARD=ARD, variance=1)
        gp0 = GPy.models.GPRegression(a[0, :].reshape(1, -1), fscalar, noise_var=0.1 ** 2, kernel=KERNEL_f)
        gp1 = GPy.models.GPRegression(a[0, :].reshape(1, -1), g1scalar, noise_var=0.1 ** 2, kernel=KERNEL_g)

        self.opt = GoSafeOptPractical(gp=[gp0, gp1], gp_full=[gp_full0, gp_full1], bounds=bounds,beta=3,
                                          fmin=[-np.inf, 0], x_0=x0.reshape(-1, 1), L_states=L_states, eta_L=0.3,eta_u=0.75,
                                          max_S1_steps=100,
                                          max_S3_steps=10, eps=0.1, max_data_size=1000, reset_size=500,
                                          boundary_thresshold_l=0.90,boundary_thresshold_u=0.94)

        y = np.array([f, g1])
        y=y.squeeze()
        self.add_data(x,y)
        self.time_recorded = []
        self.df = pd.DataFrame(
            np.array([[self.params[0], self.params[1],self.params[2],self.params[3] ,self.opt.criterion, f[0], g1[0], f[0], False]]),
            columns=['q','kappa' ,'r','a_rho' ,'criterion', 'fval', 'gval', 'fmax', 'Boundary'])
        ## Collect more policies for S_0
        self.simulate_data()
    def simulate_data(self):
        '''
        Collect more policies for S_0
        '''
        p = [5/6,1]
        d = [-0.9,-2/3]
        kappa = 0.5
        rho_action = 1
        for i in range(2):
            self.params = np.asarray([p[i],kappa,d[i],rho_action])
            f, g1,state = self.sys.simulate(self.params,update=True,opt=self.opt)
            g1 -= self.error_bound
            g1 = -g1/self.error_bound
            f -= self.mean_reward
            f /= self.std_reward
            print(f[0],g1[0])
            y = np.array([[f], [g1]])
            y = y.squeeze()
            self.add_data(state,y)
            df2= pd.DataFrame(
                np.array([[self.params[0], self.params[1], self.params[2], self.params[3], self.opt.criterion, f[0],
                           g1[0], f[0], self.sys.at_boundary]]),
                columns=['q', 'kappa', 'r', 'a_rho', 'criterion', 'fval', 'gval', 'fmax', 'Boundary'])
            self.df = self.df.append(df2)
            #self.opt.add_new_data_point(x, y)


    def optimize(self,update_boundary=False):
        '''
        run 1 full optimization step. If update_boudary=True, reevalutes boundary/failed states
        '''
        if update_boundary:
            self.opt.update_boundary_points()
        start_time = time.time()
        param = self.opt.optimize()
        self.time_recorded.append(time.time() - start_time)
        print(param, end="")
        f, g1,state = self.sys.simulate(param,update=True,opt=self.opt)
        g1 -= self.error_bound
        g1 = -g1/self.error_bound
        f -= self.mean_reward
        f /= self.std_reward
        df2 = pd.DataFrame(
            np.array([[param[0], param[1], param[2], param[3], self.opt.criterion, f[0],
                       g1[0], f[0], self.sys.at_boundary]]),
            columns=['q', 'kappa', 'r', 'a_rho', 'criterion', 'fval', 'gval', 'fmax', 'Boundary'])
        self.df = self.df.append(df2)
        y = np.array([[f], [g1]])
        y = y.squeeze()
        if not self.sys.at_boundary:
            self.add_data(state,y)
            constraint_satisified = g1[0] >= 0
            if not constraint_satisified:
                self.failure_overshoot+=-(g1[0]*self.error_bound)+self.error_bound
                logging.warning("Hit Constraint")
                print(" Hit Constraint         ",end="")
            self.failures += constraint_satisified
            print(f[0], g1[0], self.opt.criterion, constraint_satisified)
        else:
            if not self.sys.Fail:
                constraint_satisified = g1[0] >= 0
                if not constraint_satisified:
                    self.failure_overshoot += -(g1[0] * self.error_bound) + self.error_bound
                    logging.warning("Hit Constraint")
                    print(" Hit Constraint         ",g1[0], end="")
                self.opt.add_boundary_points(param)
                self.failures += constraint_satisified
            else:
                constraint_satisified=g1[0]>=0
                if not constraint_satisified:
                    logging.warning("Failed")


            print(self.opt.criterion,constraint_satisified)



    def add_data(self,state,y):
        '''
        Add data to GP
        '''
        state = np.asarray(state).squeeze()
        size = min(100, state.shape[0])
        idx = np.zeros(size + 1, dtype=int)
        idx[1:] = np.random.randint(low=1, high=len(state), size=size)
        idx[0] = 0
        state[0][self.opt.state_idx] = self.opt.x_0.squeeze().copy()
        x = state[idx]
        y = y[:, idx].T
        self.opt.add_data(x, y)








#opt=SafeOpt_Optimizer()
def experiment(method="SafeOpt"):
    #method="GoSafe"
    iterations=201
    runs=20

    if method=="GoSafeOpt":
        Reward_data=np.zeros([41,runs])
        Overshoot_summary=np.zeros([2,runs])
        for r in range(runs):
            j=0
            opt=GoSafeOpt_Optimizer()
            random.seed(r+2)
            np.random.seed(r+2)
            opt.sys.env.seed(r+2)
            opt.opt._seed(r+2)

            for i in range(iterations):
                # Collect optimum after every 5 iterations
                if i%5==0:
                    maximum, fval = opt.opt.get_maximum()
                    Reward_data[j,r]=fval[0]
                    j+=1
                update_boundary=False
                if i%30==0:
                    update_boundary=True
                opt.optimize(update_boundary=update_boundary)
                print(i)
            # Measure failures (constraint violation) during experiment
            print(opt.failures / iterations)
            Overshoot_summary[0, r] = opt.failures / iterations
            failure=np.maximum(1e-3,iterations-opt.failures)
            Overshoot_summary[1,r]=opt.failure_overshoot/(failure)
            print(opt.failure_overshoot/(failure),r)
            max, f = opt.opt.get_maximum()
            print(max,f)
            opt.df.to_csv("Optimizer_queryPoints_path" + str(r) + ".csv")
        # Save data from rewards and failure
        np.savetxt('GoSafe_error.csv', Overshoot_summary, delimiter=',')
        np.savetxt('GoSafe_Reward.csv', Reward_data, delimiter=',')

    elif method=="SafeOpt":
        
        Reward_data = np.zeros([41, runs])
        Overshoot_summary = np.zeros([2, runs])
        for r in range(runs):
            j=0
            opt = SafeOpt_Optimizer()
            random.seed(r + 2)
            np.random.seed(r + 2)
            opt.sys.env.seed(r + 2)
            for i in range(iterations):
                if i%5==0:
                    maximum, f = opt.opt.get_maximum()
                    Reward_data[j, r] = f
                    j+=1
                opt.optimize()
                print(i)
            print(opt.failures / iterations,r)

            Overshoot_summary[0, r] = opt.failures / iterations
            failure=np.maximum(1e-3,iterations-opt.failures)
            Overshoot_summary[1, r] = opt.failure_overshoot / (failure)
            max, f = opt.opt.get_maximum()
            print(max, f)
        np.savetxt('SafeOpt_Overshoot.csv', Overshoot_summary, delimiter=',')
        np.savetxt('SafeOpt_Reward.csv', Reward_data, delimiter=',')





    print(opt.failures/iterations)
    max,f=opt.opt.get_maximum()
    #max=[2.00179108e+00,4.13625539e+00, 3.34599393e+00, 7.41304209e-01,2.81500345e-01, 3.13137132e-03]
    time_recorder=np.asarray(opt.time_recorded)
    print("Time:",time_recorder.mean(),time_recorder.std())
    print("maximum",max)

    #f,g1,g2,state=opt.sys.simulate(max,update=True,render=True)



