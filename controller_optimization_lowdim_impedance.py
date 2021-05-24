import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from safeopt import GoSafeSwarm,SafeOptSwarm
import gym
import Panda_Env #Library defined for the panda environment
import mujoco_py
import scipy
from osc_controller import inverse_dynamics_control
import GPy
import random
import time
random.seed(10)
np.random.seed(10)
class System(object):

    def __init__(self,position_bound,velocity_bound,rollout_limit=0,upper_eigenvalue=0):
        self.env = gym.make("PandaEnv-v0")

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
        T1 = np.zeros(9)
        T1[4] = 1
        T1[6:] = 1
        T = np.diag(T1)
        N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
        self.N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))

    def simulate(self,params=None,render=False,opt=None,update=False):
        x0=None
        if params is not None:

            if update:
                param_a=self.set_params(params)
                self.Q = np.diag(param_a)
            else:
                self.Q=np.diag(params)

            self.R=np.eye(3)/100*np.power(10,3*params[1]) #param is between -1 and 1

            if opt is not None:
                if opt.criterion in ["S2"]:
                    x0=params[opt.state_idx]
                    x0[3:]=np.zeros(3)


        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
        K = scipy.linalg.inv(self.R) * (self.B.T * P)


        K = np.asarray(K)
        eigen_value = np.linalg.eig(self.A - np.dot(self.B, K))
        eigen_value = np.max(np.asarray(eigen_value[0]).real)

        Kp = K[:, :3]
        Kd = K[:, 3:]
        Objective=0
        self.reset(x0)
        state = []
        constraint2 = 0
        if x0 is not None:
            x=np.hstack([params[:2].reshape(1,-1),x0.reshape(1,-1)])
            state.append(x)

        else:
            obs = self.obs["observation"].copy()
            obs[:3] = obs[:3] - self.env.goal
            obs[:3] /= self.position_bound
            obs[3:] /= self.velocity_bound
            x = np.hstack([params[:2].reshape(1, -1), obs.reshape(1, -1)])
            state.append(x)


        if opt is not None:
            if eigen_value>self.upper_eigenvalue and opt.criterion=="S3":
                self.at_boundary=True
                opt.s3_steps=np.maximum(0,opt.s3_steps-1)
                return 0,0,0,state


        if render:
            init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])

            for i in range(self.T):

                bias = self.ID.g()


                J = self.ID.Jp(self.id)

                wM_des = np.dot(Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"])) - np.dot(Kd, self.obs["observation"][3:]
                                                                                                     - np.ones(3) * 1 / self.env.Tmax * (i < self.env.Tmax))
                u=-bias


                u += np.dot(J.T, wM_des)
                u=np.dot(self.N_bar,u)
                self.obs, reward, done, info = self.env.step(u)
                Objective+=reward
                constraint2=np.maximum(np.linalg.norm(self.env.goal - self.obs["achieved_goal"])-init_dist,constraint2)
                #constraint_2 = np.maximum(np.max(np.abs(self.obs["observation"][3:])), constraint_2)
                self.env.render()


        else:
            init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])

            for i in range(self.T):
                if opt is not None and not self.at_boundary:
                    obs=self.obs["observation"].copy()
                    obs[:3]-=self.env.goal
                    obs[:3]/=self.position_bound
                    obs[3:]/=self.velocity_bound
                    #obs[3:]=opt.x_0[:,3:]
                    self.at_boundary, self.Fail, params = opt.check_rollout(state=obs.reshape(1,-1), action=params)

                    if self.Fail:
                        print("Failed",i,end="")
                        return 0, 0,0,state
                    elif self.at_boundary:
                        params = params.squeeze()
                        print(" Changed action to",i,params,end="")
                        param_a = self.set_params(params.squeeze())
                        self.Q = np.diag(param_a)

                        np.eye(3) / 100 * np.power(10, 3 * params[1])
                        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
                        K = scipy.linalg.inv(self.R) * (self.B.T * P)

                        K = np.asarray(K)

                        Kp = K[:, :3]
                        Kd = K[:, 3:]


                if i < self.rollout_limit:
                    obs=self.obs["observation"].copy()
                    obs[:3]=obs[:3]-self.env.goal
                    obs[:3] /= self.position_bound
                    obs[3:] /= self.velocity_bound
                    x=np.hstack([params[:2].reshape(1,-1),obs.reshape(1,-1)])
                    state.append(x)
                bias = self.ID.g()


                J = self.ID.Jp(self.id)


                wM_des = np.dot(Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"]))-np.dot(Kd,self.obs["observation"][3:]-np.ones(3)*1/self.env.Tmax*(i<self.env.Tmax))
                u=-bias
                u += np.dot(J.T, wM_des)
                u = np.dot(self.N_bar, u)
                self.obs, reward, done, info = self.env.step(u)
                Objective+=reward
                constraint2 = np.maximum(np.linalg.norm(self.env.goal - self.obs["achieved_goal"]) - init_dist,
                                         constraint2)

                #constraint2 = np.maximum(np.max(np.abs(self.obs["observation"][3:])), constraint2)





        return Objective/self.T,constraint2,eigen_value,state


    def reset(self,x0=None):
        self.obs = self.env.reset()
        self.Fail=False
        self.at_boundary=False

        if x0 is not None:
            x0*=self.position_bound
            self.env.goal=self.obs["observation"][:3]-x0[:3]
            #Kp=500*np.eye(3)
            #Kd=np.sqrt(Kp)*0.1
            #x[:3]+=self.env.goal.squeeze()
            #for i in range(5000):
            #    bias = self.ID.g()
            #    M = self.ID.M()

            #    J = self.ID.Jp(self.id)

            #    Mx, Mx_inv = self.ID.Mx(M, J)
            #    wM_des = np.dot(Kp, (x[:3] - self.obs["achieved_goal"])) - np.dot(Kd, self.obs["observation"][3:]-x[3:])
            #    u = -bias
            #    u += np.dot(J.T, np.dot(Mx, wM_des))
            #    self.obs, reward, done, info = self.env.step(u)

            #print(self.obs["observation"]-x)




    def set_params(self, params):
        q1 = np.repeat(np.power(10, 6*params[0]),3) #param is between -1 and 1
        q2 = np.sqrt(q1)*0.1
        updated_params = np.hstack((q1.squeeze(), q2.squeeze()))
        return updated_params


class SafeOpt_Optimizer(object):
    def __init__(self, upper_overshoot=0.05,upper_eigenvalue=-10, lengthscale=0.65, ARD=True):
        self.upper_eigenvalue=upper_eigenvalue
        self.upper_overshoot=upper_overshoot
        q=4/6
        r=-1
        self.params = np.asarray([q, r])
        self.failures = 0
        self.mean_reward = -0.33
        self.std_reward = 0.14
        self.eigen_value_std = 21
        self.sys = System(rollout_limit=0, position_bound=0.5,
                          velocity_bound=7,
                          upper_eigenvalue=self.upper_eigenvalue)

        f, g1, g2, state = self.sys.simulate(self.params, update=True)
        g2 = self.upper_eigenvalue - g2
        g2 /= self.eigen_value_std
        g1 -= self.upper_overshoot
        g1 = -g1 / self.upper_overshoot
        f -= self.mean_reward
        f /= self.std_reward
        print(f, g1, g2)
        f = np.asarray([[f]])
        f = f.reshape(-1, 1)
        g1 = np.asarray([[g1]])
        g1 = g1.reshape(-1, 1)
        L = [lengthscale / 6, lengthscale / 3]
        x=self.params.reshape(1,-1)
        KERNEL_f = GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        # KERNEL_f = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        KERNEL_g = GPy.kern.sde_Matern52(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        # KERNEL_g = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        gp0 = GPy.models.GPRegression(x[0, :].reshape(1, -1), f, noise_var=0.1 ** 2, kernel=KERNEL_f)
        gp1 = GPy.models.GPRegression(x[0, :].reshape(1, -1), g1, noise_var=0.1 ** 2, kernel=KERNEL_g)

        bounds = [[1 / 3, 1], [-1, 1]]

        self.opt = SafeOptSwarm([gp0, gp1], fmin=[-np.inf, 0], bounds=bounds, beta=4)
        self.time_recorded = []
        self.simulate_data()

    def simulate_data(self):
        p = [5/6,1]
        d = [-0.9,-2/3]

        for i in range(1):
            self.params = np.asarray([p[i],d[i]])
            f, g1,g2,state = self.sys.simulate(self.params,update=True)
            g2 = self.upper_eigenvalue - g2
            g2 /= self.eigen_value_std
            g1 -= self.upper_overshoot
            g1 = -g1/self.upper_overshoot
            f -= self.mean_reward
            f /= self.std_reward
            print(f,g1,g2)
            y = np.array([[f], [g1]])
            y = y.squeeze()
            self.opt.add_new_data_point(self.params.reshape(1, -1), y)


    def optimize(self):
        start_time = time.time()
        param = self.opt.optimize()
        self.time_recorded.append(time.time() - start_time)
        print(param, end="")
        f, g1,g2,state = self.sys.simulate(param,update=True)
        #print(f, g1, g2,self.opt.criterion)
        g2 = self.upper_eigenvalue - g2
        g2 /= self.eigen_value_std
        g1 -= self.upper_overshoot
        g1 = -g1/self.upper_overshoot
        f -= self.mean_reward
        f /= self.std_reward

        y = np.array([[f], [g1]])
        y = y.squeeze()
        self.opt.add_new_data_point(param.reshape(1, -1), y)
        constraint_satisified = g1 >= 0 and g2 >= 0
        self.failures += constraint_satisified
        print(f, g1, g2, constraint_satisified)





class GoSafe_Optimizer(object):

    def __init__(self, upper_overshoot=0.05,upper_eigenvalue=-10, lengthscale=0.65, ARD=True):
        self.upper_eigenvalue=upper_eigenvalue
        self.upper_overshoot=upper_overshoot
        q =4/6
        r = -1
        self.params = np.asarray([q,r])
        self.failures=0
        self.rollout_limit = 500
        self.mean_reward = -0.33
        self.std_reward = 0.14
        self.eigen_value_std=21
        self.position_bound = 0.5
        self.velocity_bound = 7
        self.sys = System(rollout_limit=self.rollout_limit,position_bound=self.position_bound,velocity_bound=self.velocity_bound,
                          upper_eigenvalue=self.upper_eigenvalue)
        f, g1,g2, state = self.sys.simulate(self.params,update=True)
        g2=self.upper_eigenvalue-g2
        g2/=self.eigen_value_std
        g1-=self.upper_overshoot
        g1=-g1/self.upper_overshoot
        f -= self.mean_reward
        f /= self.std_reward
        print(f, g1,g2)
        f = np.asarray([[f]])
        f = f.reshape(-1, 1)
        g1 = np.asarray([[g1]])
        g1 = g1.reshape(-1, 1)
        #g2 = np.asarray([[g2]])
        #g2 = g2.reshape(-1, 1)


        x0=state[0][:,2:]


        x=np.asarray(state).squeeze()


        L=[lengthscale/6,lengthscale/3,0.5,0.5,0.5,20/7,20/7,20/7]
        KERNEL_f = GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        #KERNEL_f = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        KERNEL_g = GPy.kern.sde_Matern52(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        #KERNEL_g = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
        gp0 = GPy.models.GPRegression(x[0,:].reshape(1,-1), f, noise_var=0.1 ** 2, kernel=KERNEL_f)
        gp1 = GPy.models.GPRegression(x[0,:].reshape(1,-1), g1, noise_var=0.1 ** 2, kernel=KERNEL_g)
        #gp2 = GPy.models.GPRegression(x[0, :].reshape(1, -1), g2, noise_var=0.01 ** 2, kernel=KERNEL_g)





        bounds = [[1/3, 1], [-1, 1],
                           [-1, 1],[-1, 1],[-1, 1],[-1,1],[-1, 1],[-1, 1]]


        self.opt = GoSafeSwarm([gp0,gp1], fmin=[-np.inf, 0], bounds=bounds, beta=4,x_0=x0.reshape(-1,1),eta=0.1,tol=0.05,max_S2_steps=10,max_S1_steps=30,max_S3_steps=10,eps=0.1,max_expansion_steps=100,reset_size=400,max_data_size=800)
        self.opt.S3_x0_ratio=1
        #self.opt.safety_cutoff=0.9
        y = np.array([f, g1])
        y=y.squeeze()
        self.add_data(x,y)
        #for i in range(1,self.rollout_limit):
         #   self.opt.add_new_data_point(x[i].reshape(1,-1),y)
        self.time_recorded = []
        self.simulate_data()
    def simulate_data(self):
        p = [5/6,1]
        d = [-0.9,-2/3]

        for i in range(1):
            self.params = np.asarray([p[i],d[i]])
            f, g1,g2,state = self.sys.simulate(self.params,update=True,opt=self.opt)
            g2 = self.upper_eigenvalue - g2
            g2 /= self.eigen_value_std
            g1 -= self.upper_overshoot
            g1 = -g1/self.upper_overshoot
            f -= self.mean_reward
            f /= self.std_reward
            print(f,g1,g2)
            y = np.array([[f], [g1]])
            y = y.squeeze()
            self.add_data(state,y)
            #self.opt.add_new_data_point(x, y)


    def optimize(self):
        start_time = time.time()
        param = self.opt.optimize()
        self.time_recorded.append(time.time() - start_time)
        print(param, end="")
        f, g1,g2,state = self.sys.simulate(param,update=True,opt=self.opt)
        #print(f, g1, g2,self.opt.criterion)
        g2 = self.upper_eigenvalue - g2
        g2 /= self.eigen_value_std
        g1 -= self.upper_overshoot
        g1 = -g1/self.upper_overshoot
        f -= self.mean_reward
        f /= self.std_reward

        y = np.array([[f], [g1]])
        y = y.squeeze()
        if not self.sys.at_boundary:
            self.add_data(state,y)
            constraint_satisified = g1 >= 0 and g2>=0
            self.failures += constraint_satisified
            print(f, g1,g2, self.opt.criterion, constraint_satisified)
        else:
            if not self.sys.Fail:
                constraint_satisified = 1
                self.failures += constraint_satisified
            self.opt.add_boundary_points(param)
            print(self.opt.criterion)



    def add_data(self,state,y):
        state=np.asarray(state).squeeze()
        size=min(50,state.shape[0])
        idx=np.zeros(size+1,dtype=int)
        idx[1:]=np.random.randint(low=1,high=len(state),size=size)
        idx[0]=0
        for i in idx:
            x=state[i]
            self.opt.add_new_data_point(x.reshape(1, -1), y)


#opt=SafeOpt_Optimizer()
method="SafeOpt"
iterations=201
if method=="GoSafe":
    opt=GoSafe_Optimizer()

    for i in range(iterations):
        if i%20==0:
            q=np.linspace(-1,1,25)
            r=np.linspace(-1,1,25)
            a = np.asarray(np.meshgrid(q, r)).T.reshape(-1, 2)

            input=np.zeros([a.shape[0],2+opt.opt.state_dim])
            input[:,2:]=opt.opt.x_0
            input[:,:2]=a
            mean, var = opt.opt.gps[1].predict(input)
            std=np.sqrt(var)
            l_x0 = mean -opt.opt.beta(opt.opt.t)*std
            safe_idx=np.where(l_x0>=0)[0]
            values=np.zeros(a.shape[0])
            values[safe_idx]=1

            mean, var = opt.opt.gps[0].predict(input)
            l_f = mean - opt.opt.beta(opt.opt.t) * std

            safe_l_f=l_f[safe_idx]
            safe_max=np.where(l_f==safe_l_f.max())[0]
            optimum_params=a[safe_max,:]
            optimum_params=optimum_params.squeeze()
            q=np.reshape(a[:,0],[25,25])
            r = np.reshape(a[:, 1], [25, 25])
            values = values.reshape([25, 25])
            colours = ['red', 'green']
            fig = plt.figure(figsize=(10, 10))
            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = fig.add_axes([left, bottom, width, height])
            ax.set_xlabel('q')
            ax.set_ylabel('r')
            cs = ax.contourf(q*6, r*3, values)
            ax.scatter(q*6, r*3, c=values, cmap=matplotlib.colors.ListedColormap(colours))
            ax.scatter(optimum_params[0]*6, optimum_params[1]*3, marker="<", color="b", s=np.asarray([200]))
            ax.set_title("Safe Set Belief, iter "+str(i))
            ax.set_ylim([-3.1, 3.1])
            ax.set_xlim([-6.1, 6.1])

            plt.savefig('Safeset' + str(i) +'.png', dpi=300)

        opt.optimize()
        print(i)

elif method=="SafeOpt":
    opt=SafeOpt_Optimizer()

    for i in range(iterations):
        if i%20==0:
            q=np.linspace(-1,1,25)
            r=np.linspace(-1,1,25)
            a = np.asarray(np.meshgrid(q, r)).T.reshape(-1, 2)
            input=a
            mean, var = opt.opt.gps[1].predict(input)
            std=np.sqrt(var)
            l_x0 = mean -opt.opt.beta(opt.opt.t)*std
            safe_idx=np.where(l_x0>=0)[0]
            values=np.zeros(a.shape[0])
            values[safe_idx]=1

            mean, var = opt.opt.gps[0].predict(input)
            l_f = mean - opt.opt.beta(opt.opt.t) * std

            safe_l_f=l_f[safe_idx]
            safe_max=np.where(l_f==safe_l_f.max())[0]
            optimum_params=a[safe_max,:]
            optimum_params=optimum_params.squeeze()
            q=np.reshape(a[:,0],[25,25])
            r = np.reshape(a[:, 1], [25, 25])
            values = values.reshape([25, 25])
            colours = ['red', 'green']
            fig = plt.figure(figsize=(10, 10))
            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = fig.add_axes([left, bottom, width, height])
            ax.set_xlabel('q')
            ax.set_ylabel('r')
            cs = ax.contourf(q*6, r*3, values)
            ax.scatter(q*6, r*3, c=values, cmap=matplotlib.colors.ListedColormap(colours))
            ax.scatter(optimum_params[0]*6, optimum_params[1]*3, marker="<", color="b", s=np.asarray([200]))
            ax.set_title("Safe Set Belief, iter "+str(i))
            ax.set_ylim([-3.1, 3.1])
            ax.set_xlim([-6.1, 6.1])

            plt.savefig('Safeset' + str(i) +'.png', dpi=300)

        opt.optimize()
        print(i)



print(opt.failures/iterations)
max,f=opt.opt.get_maximum()
#max=[2.00179108e+00,4.13625539e+00, 3.34599393e+00, 7.41304209e-01,2.81500345e-01, 3.13137132e-03]
time_recorder=np.asarray(opt.time_recorded)
print("Time:",time_recorder.mean(),time_recorder.std())
print("maximum",max)

f,g1,g2,state=opt.sys.simulate(max,update=True,render=True)




