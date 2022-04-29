import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from gosafeopt import SafeOptSwarm, GoSafeOptPractical
import gym
import pandaenv  # Library defined for the panda environment
import mujoco_py
import scipy
from pandaenv.utils import inverse_dynamics_control
import GPy
import random
import time
import logging
plt.rcParams.update({'font.size': 16})
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
class System(object):

    def __init__(self, position_bound, velocity_bound, rollout_limit=0, upper_eigenvalue=0):
        self.env = gym.make("PandaEnvBasic-v0")
        self.low_eigen_value = False
        # Define Q,R, A, B, C matrices
        self.Q = np.eye(6)
        self.R = np.eye(3) / 100
        self.env.seed(0)
        self.obs = self.env.reset()
        self.A = np.zeros([6, 6])
        # A=np.zeros([18,18])
        self.A[:3, 3:] = np.eye(3)
        self.B = np.zeros([6, 3])
        self.B[3:, :] = np.eye(3)
        self.T = 2000
        # Set up inverse dynmaics controller
        self.ID = inverse_dynamics_control(env=self.env, njoints=9, target=self.env.goal)
        self.id = self.env.sim.model.site_name2id("panda:grip")
        self.rollout_limit = rollout_limit
        self.at_boundary = False
        self.Fail = False
        self.approx = True
        self.position_bound = position_bound
        self.velocity_bound = velocity_bound
        self.upper_eigenvalue = upper_eigenvalue
        # Define weighting matrix to set torques 4,6,7,.. (not required for movement) to 0.
        T1 = np.zeros(9)
        T1[4] = 1
        T1[6:] = 1
        T = np.diag(T1)
        N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
        self.N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))

    def simulate(self, params=None, render=False, opt=None, update=False):
        # Simulate the robot/run experiments
        x0 = None
        # Update parameters
        if params is not None:

            if update:
                param_a = self.set_params(params)
                self.Q = np.diag(param_a)
            else:
                self.Q = np.diag(params)

            self.R = np.eye(3) / 100 * np.power(10, 3 * params[1])  # param is between -1 and 1

            if opt is not None:
                # Update initial condition if running GoSafe
                if opt.criterion in ["S2"]:
                    x0 = params[opt.state_idx]
                    x0 = x0[:-1]
                    x0[3:] = np.zeros(3)
                    
        # Get controller
        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
        K = scipy.linalg.inv(self.R) * (self.B.T * P)

        K = np.asarray(K)
        eigen_value = np.linalg.eig(self.A - np.dot(self.B, K))
        eigen_value = np.max(np.asarray(eigen_value[0]).real)

        Kp = K[:, :3]
        Kd = K[:, 3:]
        Objective = 0
        self.reset(x0)
        state = []
        constraint2 = 0
        init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
        # dist=init_dist-self.init_dist
        # dist=np.asarray(dist).reshape(1,-1)
        if x0 is not None:
            x = np.hstack([params[:2].reshape(1, -1), x0.reshape(1, -1)])
            state.append(x)

        else:
            obs = self.obs["observation"].copy()
            obs[:3] = obs[:3] - self.env.goal
            obs[:3] /= self.position_bound
            obs[3:] /= self.velocity_bound
            x = np.hstack([params[:2].reshape(1, -1), obs.reshape(1, -1)])
            state.append(x)

        if opt is not None:
            if eigen_value > self.upper_eigenvalue and opt.criterion == "S3":
                self.at_boundary = True
                self.low_eigen_value = True
                opt.s3_steps = np.maximum(0, opt.s3_steps - 1)
                return np.zeros(1), np.zeros(1), 0, state

        elif eigen_value > self.upper_eigenvalue:
            self.at_boundary = True
            self.low_eigen_value = True
            self.Fail = True
            print("Eigenvalues too high ", end="")
            return np.zeros(1), np.zeros(1), np.zeros(1), state

        if render:
            # init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
            # init_dist=self.init_dist
            for i in range(self.T):
                bias = self.ID.g()

                J = self.ID.Jp(self.id)

                wM_des = np.dot(Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"])) - np.dot(Kd, self.obs[
                                                                                                             "observation"][
                                                                                                         3:]
                                                                                                     - np.ones(
                    3) * 1 / self.env.Tmax * (i < self.env.Tmax))
                u = -bias

                u += np.dot(J.T, wM_des)
                u = np.dot(self.N_bar, u)
                self.obs, reward, done, info = self.env.step(u)
                Objective += reward
                constraint2 = np.maximum(np.linalg.norm(self.env.goal - self.obs["achieved_goal"]) - init_dist,
                                         constraint2)
                # constraint_2 = np.maximum(np.max(np.abs(self.obs["observation"][3:])), constraint_2)
                self.env.render()


        else:
            # Simulate the arm
            # init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
            # init_dist = self.init_dist
            constraint2 = np.zeros(self.rollout_limit + 1)
            Objective = np.zeros(self.rollout_limit + 1)
            for i in range(self.T):
                if opt is not None and not self.at_boundary:
                    obs = self.obs["observation"].copy()
                    obs[:3] -= self.env.goal
                    obs[:3] /= self.position_bound
                    obs[3:] /= self.velocity_bound
                    # obs[3:]=opt.x_0[:,3:]
                    # Evaluate boundary condition
                    if i % 1 == 0:
                        # x_check=np.hstack([])
                        x_check = obs.reshape(1, -1)
                        self.at_boundary, self.Fail, params = opt.check_rollout(state=x_check, action=params)

                    if self.Fail:
                        print("FAILED                  ", i, end=" ")
                        return np.zeros(1), np.zeros(1), 0, state
                    elif self.at_boundary:
                        params = params.squeeze()
                        print(" Changed action to", i, params, end="")
                        param_a = self.set_params(params.squeeze())
                        self.Q = np.diag(param_a)

                        self.R = np.eye(3) / 100 * np.power(10, 3 * params[1])
                        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
                        K = scipy.linalg.inv(self.R) * (self.B.T * P)

                        K = np.asarray(K)

                        Kp = K[:, :3]
                        Kd = K[:, 3:]
                # Collect rollouts
                if i < self.rollout_limit:
                    obs = self.obs["observation"].copy()
                    obs[:3] = obs[:3] - self.env.goal
                    obs[:3] /= self.position_bound
                    obs[3:] /= self.velocity_bound
                    x = np.hstack([params[:2].reshape(1, -1), obs.reshape(1, -1)])
                    state.append(x)
                    constraint2[i] = 0
                    Objective[i] = 0
                bias = self.ID.g()

                J = self.ID.Jp(self.id)

                wM_des = np.dot(Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"])) - np.dot(Kd, self.obs[
                                                                                                             "observation"][
                                                                                                         3:] - np.ones(
                    3) * 1 / self.env.Tmax * (i < self.env.Tmax))
                u = -bias
                u += np.dot(J.T, wM_des)
                u = np.dot(self.N_bar, u)
                self.obs, reward, done, info = self.env.step(u)
                Objective += reward
                constraint2 = np.maximum(
                    (np.linalg.norm(self.env.goal - self.obs["achieved_goal"]) - init_dist) * np.ones(
                        self.rollout_limit + 1),
                    constraint2)

                # constraint2 = np.maximum(np.max(np.abs(self.obs["observation"][3:])), constraint2)

        return Objective / self.T, constraint2 / init_dist, eigen_value, state

    def reset(self, x0=None):
        '''
        Reset environment for next experiment
        '''
        self.obs = self.env.reset()
        self.init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
        self.Fail = False
        self.at_boundary = False
        self.low_eigen_value = False
        if x0 is not None:
            x0 *= self.position_bound
            self.env.goal = self.obs["observation"][:3] - x0[:3]
            # Kp=500*np.eye(3)
            # Kd=np.sqrt(Kp)*0.1
            # x[:3]+=self.env.goal.squeeze()
            # for i in range(5000):
            #    bias = self.ID.g()
            #    M = self.ID.M()

            #    J = self.ID.Jp(self.id)

            #    Mx, Mx_inv = self.ID.Mx(M, J)
            #    wM_des = np.dot(Kp, (x[:3] - self.obs["achieved_goal"])) - np.dot(Kd, self.obs["observation"][3:]-x[3:])
            #    u = -bias
            #    u += np.dot(J.T, np.dot(Mx, wM_des))
            #    self.obs, reward, done, info = self.env.step(u)

            # print(self.obs["observation"]-x)

    def set_params(self, params):
        '''
        Update parameters for controller
        '''
        q1 = np.repeat(np.power(10, 6 * params[0]), 3)  # param is between -1 and 1
        q2 = np.sqrt(q1) * 0.1
        updated_params = np.hstack((q1.squeeze(), q2.squeeze()))
        return updated_params


class GoSafe_Optimizer(object):
    '''
    Optimizer
    '''
    def __init__(self, upper_overshoot=0.08, upper_eigenvalue=-10, lengthscale=0.7, ARD=True):
        self.upper_eigenvalue = upper_eigenvalue
        self.upper_overshoot = upper_overshoot
        q = 4 / 6
        r = -1
        self.params = np.asarray([q, r])
        self.failures = 0
        self.failure_overshoot = 0
        #self.rollout_limit = 250
        self.rollout_limit=500
        self.mean_reward = -0.33
        self.std_reward = 0.14
        self.eigen_value_std = 21
        self.position_bound = 0.5
        self.velocity_bound = 7
        # Set up system and optimizer
        self.sys = System(rollout_limit=self.rollout_limit, position_bound=self.position_bound,
                          velocity_bound=self.velocity_bound,
                          upper_eigenvalue=self.upper_eigenvalue)
        f, g1, g2, state = self.sys.simulate(self.params, update=True)
        g2 = self.upper_eigenvalue - g2
        g2 /= self.eigen_value_std
        g1 -= self.upper_overshoot
        g1 = -g1 / self.upper_overshoot
        f -= self.mean_reward
        f /= self.std_reward
        print(f[0], g1[0], g2)
        fscalar = f[0]
        g1scalar = g1[0]
        fscalar = np.asarray([[fscalar]])
        fscalar = fscalar.reshape(-1, 1)
        g1scalar = np.asarray([[g1scalar]])
        g1scalar = g1scalar.reshape(-1, 1)


        x0 = state[0][:, 2:]

        x = np.asarray(state).squeeze()

        L = [lengthscale / 6, lengthscale / 6, 0.3, 0.3, 0.3, 2.5, 2.5, 2.5]
        L_states = np.asarray(L[2:])

        KERNEL_f = GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)

        KERNEL_g = GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        
        gp_full0 = GPy.models.GPRegression(x[0, :].reshape(1, -1), fscalar, noise_var=0.1 ** 2, kernel=KERNEL_f)
        gp_full1 = GPy.models.GPRegression(x[0, :].reshape(1, -1), g1scalar, noise_var=0.1 ** 2, kernel=KERNEL_g)

        a=x[:,:2]
        L = np.asarray(L[:2])
 
        KERNEL_f= GPy.kern.sde_Matern32(input_dim=a.shape[1], lengthscale=L, ARD=ARD, variance=1)
 
        KERNEL_g = GPy.kern.sde_Matern32(input_dim=a.shape[1], lengthscale=L, ARD=ARD, variance=1)

        gp0 = GPy.models.GPRegression(a[0, :].reshape(1, -1), fscalar, noise_var=0.1 ** 2, kernel=KERNEL_f)
        gp1 = GPy.models.GPRegression(a[0, :].reshape(1, -1), g1scalar, noise_var=0.1 ** 2, kernel=KERNEL_g)

        bounds = [[1 / 3, 1], [-1, 1]]


        self.opt = GoSafeOptPractical(gp=[gp0, gp1], gp_full=[gp_full0, gp_full1], bounds=bounds, beta=4.0,
                                          fmin=[-np.inf, 0], x_0=x0.reshape(-1, 1), L_states=L_states, eta_L=0.4,
                                          eta_u=0.6, max_S1_steps=30,
                                          max_S3_steps=10, eps=0.1, max_data_size=1000, reset_size=500,
                                          boundary_thresshold_l=0.90, boundary_thresshold_u=0.94)
        
        y = np.array([f, g1])
        y = y.squeeze()
        self.add_data(x, y)
  
        self.time_recorded = []
        self.df = pd.DataFrame(
            np.array([[self.params[0], self.params[1], self.opt.criterion, f[0], g1[0], f[0], False]]),
            columns=['q', 'r', 'criterion', 'fval', 'gval', 'fmax', 'Boundary'])
        # Collect more initial policies for S_0
        self.simulate_data()

    def simulate_data(self):
        '''
        Collect more initial policies
        '''
        p = [5 / 6, 1]
        d = [-0.9, -2 / 3]

        for i in range(2):
            self.params = np.asarray([p[i], d[i]])
            f, g1, g2, state = self.sys.simulate(self.params, update=True, opt=self.opt)

            max, fmax = self.opt.get_maximum()

            g2 = self.upper_eigenvalue - g2
            g2 /= self.eigen_value_std
            g1 -= self.upper_overshoot
            g1 = -g1 / self.upper_overshoot
            f -= self.mean_reward
            f /= self.std_reward
            print(f[0], g1[0], g2)
            y = np.array([[f], [g1]])
            y = y.squeeze()
            self.add_data(state, y)
            df2 = pd.DataFrame(np.array(
                [[self.params[0], self.params[1], self.opt.criterion, f[0], g1[0], fmax[0], self.sys.at_boundary]]),
                               columns=['q', 'r', 'criterion', 'fval', 'gval', 'fmax', 'Boundary'])
            self.df = self.df.append(df2)

            # self.opt.add_new_data_point(x, y)

    def optimize(self, update_boundary=False):
        '''
        Performs 1 optimization step, i.e. recommend a parameter and run experiments with it
        '''
        if update_boundary:
            self.opt.update_boundary_points()
        start_time = time.time()
        param = self.opt.optimize()
        self.time_recorded.append(time.time() - start_time)
        print(param, end="")
        f, g1, g2, state = self.sys.simulate(param, update=True, opt=self.opt)
        max, fmax = opt.opt.get_maximum()
        # data = [{'q': param[0], 'r': param[1], 'criterion': self.opt.criterion, 'fval': f, 'gval': g1,
        #        'fmax': fmax}]

        # print(f, g1, g2,self.opt.criterion)
        g2 = self.upper_eigenvalue - g2
        g2 /= self.eigen_value_std
        g1 -= self.upper_overshoot
        g1 = -g1 / self.upper_overshoot
        f -= self.mean_reward
        f /= self.std_reward

        y = np.array([[f], [g1]])
        y = y.squeeze()
        df2 = pd.DataFrame(
            np.array([[param[0], param[1], self.opt.criterion, f[0], g1[0], fmax[0], self.sys.at_boundary]]),
            columns=['q', 'r', 'criterion', 'fval', 'gval', 'fmax', 'Boundary'])
        self.df = self.df.append(df2)
        if not self.sys.at_boundary:
            self.add_data(state, y)
            constraint_satisified = g1[0] >= 0
            if not constraint_satisified:
                self.failure_overshoot += -(g1[0] * self.upper_overshoot) + self.upper_overshoot
                logging.warning("Hit Constraint")
                print(" Hit Constraint         ", end="")
            self.failures += constraint_satisified
            print(f[0], g1[0], g2, self.opt.criterion, constraint_satisified)
        else:
            if not self.sys.Fail:
                constraint_satisified = g1[0] >= 0
                if not constraint_satisified:
                    self.failure_overshoot += -(g1[0] * self.upper_overshoot) + self.upper_overshoot
                    logging.warning("Hit Constraint")
                    print(" Hit Constraint         ", g1[0], end="")
                self.opt.add_boundary_points(param)
                if self.sys.low_eigen_value:
                    fake_state = self.opt.x_0.copy() + 100
                    print("Added fake state")
                    self.opt.Failed_state_list.append(fake_state)
                self.failures += constraint_satisified
            else:
                constraint_satisified = g1 >= 0
                if not constraint_satisified:
                    logging.warning("Failed")

            print(self.opt.criterion, constraint_satisified)

    def add_data(self, state, y):
        state = np.asarray(state).squeeze()
        size = min(50, state.shape[0])
        idx = np.zeros(size + 1, dtype=int)
        idx[1:] = np.random.randint(low=1, high=len(state), size=size)
        idx[0] = 0
        state[0][self.opt.state_idx] = self.opt.x_0.squeeze().copy()
        x = state[idx]
        y = y[:, idx].T
        self.opt.add_data(x, y)
        # for i in idx:
        #    x=state[i]
        #    self.opt.add_new_data_point(x.reshape(1, -1), y[:,i])


iterations = 201
runs = 20
# Ploting of safe set
plot = True
Reward_data = np.zeros([41, runs])
Overshoot_summary = np.zeros([2, runs])
for r in range(runs):
    j = 0
    opt = GoSafe_Optimizer()
    random.seed(r)
    np.random.seed(r)
    opt.sys.env.seed(r)
    opt.opt._seed(r)
    if r > 0:
        # Stop ploting safe set after 1 full run
        plot = False
    for i in range(iterations):
        if i % 5 == 0:
            # Record optimum found after every 5 iterations
            maximum, fval = opt.opt.get_maximum()
            Reward_data[j, r] = fval[0]
            j += 1
            if plot and i % 20 == 0:
                # Plot safe set after every 20 iterations
                
                # Define parameter space and determine lower bounds
                q = np.linspace(-1, 1, 25)
                r_cost = np.linspace(-1, 1, 25)
                a = np.asarray(np.meshgrid(q, r_cost)).T.reshape(-1, 2)

                input = np.zeros([a.shape[0], 2 + opt.opt.state_dim])
                input[:, 2:] = opt.opt.x_0
                input[:, :2] = a
                mean, var = opt.opt.gps[1].predict(input)
                std = np.sqrt(var)
                l_x0 = mean - opt.opt.beta(opt.opt.t) * std
                safe_idx = np.where(l_x0 >= 0)[0]
                values = np.zeros(a.shape[0])
                values[safe_idx] = 1

                mean, var = opt.opt.gps[0].predict(input)
                l_f = mean - opt.opt.beta(opt.opt.t) * std

                safe_l_f = l_f[safe_idx]
                safe_max = np.where(l_f == safe_l_f.max())[0]
                optimum_params = a[safe_max, :]
                optimum_params = optimum_params.squeeze()
                q = np.reshape(a[:, 0], [25, 25])
                r_cost = np.reshape(a[:, 1], [25, 25])
                values = values.reshape([25, 25])
                colours = ['red', 'green']
                fig = plt.figure(figsize=(10, 10))
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                ax = fig.add_axes([left, bottom, width, height])
                ax.set_xlabel(r'$q_c$')
                ax.set_ylabel(r'$r$')
                cs = ax.contourf(q * 6, r_cost * 3, values)
                ax.scatter(q * 6, r_cost * 3, c=values, cmap=matplotlib.colors.ListedColormap(colours))
                ax.scatter(optimum_params[0] * 6, optimum_params[1] * 3, marker="<", color="b", s=np.asarray([200]))
                ax.set_title("Safe Set Belief, iter " + str(i))
                ax.set_ylim([-3.1, 3.1])
                ax.set_xlim([-6.1, 6.1])

                plt.savefig('Safeset' + str(i) + '.png', dpi=300)

        
        # Optimization step
        
        # Check failed set after every 10 iterations for update
        update_boundary = False
        if i % 10 == 0:
            update_boundary = True
        opt.optimize(update_boundary=update_boundary)
        print(i)
    # Measure failures (constraint violation) and magnitude of violation
    print(opt.failures / iterations)
    Overshoot_summary[0, r] = opt.failures / iterations
    failure = np.maximum(1e-3, iterations - opt.failures)
    Overshoot_summary[1, r] = opt.failure_overshoot / (failure)
    print(opt.failure_overshoot / (failure))
    max, f = opt.opt.get_maximum()
    print(max, f)

    opt.df.to_csv("Optimizer_queryPoints" + str(r) + ".csv")
# Save recorded data on rewards and constraint violation
np.savetxt('GoSafe_Overshoot.csv', Overshoot_summary, delimiter=',')
np.savetxt('GoSafe_Reward.csv', Reward_data, delimiter=',')



print(opt.failures / iterations)
max, f = opt.opt.get_maximum()
# max=[2.00179108e+00,4.13625539e+00, 3.34599393e+00, 7.41304209e-01,2.81500345e-01, 3.13137132e-03]
time_recorder = np.asarray(opt.time_recorded)
print("Time:", time_recorder.mean(), time_recorder.std())
print("maximum", max)

# f,g1,g2,state=opt.sys.simulate(max,update=True,render=True)





