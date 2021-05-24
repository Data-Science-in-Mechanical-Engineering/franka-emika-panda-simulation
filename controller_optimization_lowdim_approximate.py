import numpy as np
import matplotlib.pyplot as plt
from safeopt import GoSafeSwarm,SafeOptSwarm
import gym
import Panda_Env #Library defined for the panda environment
import mujoco_py
import scipy
from osc_controller import inverse_dynamics_control
import GPy
class System(object):

    def __init__(self,rollout_limit=0):
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

            self.R=np.eye(3)/100*params[1]

            if opt is not None and opt.criterion in ["S2","S3"]:
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
        if x0 is not None:
            x=np.hstack([params[:2].reshape(1,-1),x0.reshape(1,-1)])
            state.append(x)

        if render:
            init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
            dist_factor = np.linalg.norm(self.env.goal - self.obs["achieved_goal"]) / init_dist
            dist_con = (1 + np.exp(-dist_factor)) / 2
            vel_con = (1 + np.exp(-0.5 * np.tanh(np.linalg.norm(self.obs["observation"][3:])))) / 2
            constraint1 = vel_con * dist_con
            constraint2=0
            for i in range(self.T):

                bias = self.ID.g()
                M = self.ID.M()

                J = self.ID.Jp(self.id)

                Mx,Mx_inv=self.ID.Mx(M,J)
                wM_des = np.dot(Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"])) - np.dot(Kd, self.obs["observation"][3:])
                u=-bias

                if self.approx:

                    diag_M = Mx.diagonal()

                    approx_M = np.diag(diag_M)

                    u += np.dot(J.T, np.dot(approx_M, wM_des))

                else:

                    u += np.dot(J.T, np.dot(Mx, wM_des))
                u=np.dot(self.N_bar,u)
                self.obs, reward, done, info = self.env.step(u)
                Objective+=reward
                dist_factor = np.linalg.norm(self.env.goal - self.obs["achieved_goal"]) / init_dist
                dist_con = (1 + np.exp(-dist_factor)) / 2
                vel_con = (1 + np.exp(-0.5 * np.tanh(np.linalg.norm(self.obs["observation"][3:])))) / 2
                constraint1 = np.maximum(constraint1,vel_con * dist_con)
                constraint2=np.maximum(np.linalg.norm(self.env.goal - self.obs["achieved_goal"])-init_dist,constraint2)
                self.env.render()


        else:
            init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
            dist_factor = np.linalg.norm(self.env.goal - self.obs["achieved_goal"]) / init_dist
            dist_con = (1 + np.exp(-dist_factor)) / 2
            vel_con = (1 + np.exp(-0.5 * np.tanh(np.linalg.norm(self.obs["observation"][3:])))) / 2
            constraint1 = vel_con * dist_con
            constraint2 = 0
            for i in range(self.T):
                if opt is not None and not self.at_boundary:
                    obs=self.obs["observation"].copy()
                    obs[:3]-=self.env.goal
                    #obs[3:]=opt.x_0[:,3:]
                    self.at_boundary, self.Fail, params = opt.check_rollout(state=obs.reshape(1,-1), action=params)

                    if self.Fail:
                        print("Failed",i,end="")
                        return None, None,None,state
                    elif self.at_boundary:
                        params = params.squeeze()
                        print(" Changed action to",i,params,end="")
                        param_a = self.set_params(params.squeeze())
                        self.Q = np.diag(param_a)

                        self.R=np.eye(3)/100*params[1]
                        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
                        K = scipy.linalg.inv(self.R) * (self.B.T * P)

                        K = np.asarray(K)

                        Kp = K[:, :3]
                        Kd = K[:, 3:]


                if i < self.rollout_limit:
                    obs=self.obs["observation"].copy()
                    obs[:3]=obs[:3]-self.env.goal
                    x=np.hstack([params[:2].reshape(1,-1),obs.reshape(1,-1)])
                    state.append(x)
                bias = self.ID.g()
                M = self.ID.M()

                J = self.ID.Jp(self.id)

                Mx,Mx_inv=self.ID.Mx(M,J)
                wM_des = np.dot(Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"]))-np.dot(Kd,self.obs["observation"][3:]-np.ones(3)*1/self.env.Tmax*(i<self.env.Tmax))
                u=-bias
                u += np.dot(J.T, np.dot(Mx, wM_des))
                u = np.dot(self.N_bar, u)
                self.obs, reward, done, info = self.env.step(u)
                Objective+=reward
                dist_factor = np.linalg.norm(self.env.goal - self.obs["achieved_goal"]) / init_dist
                dist_con = (1 + np.exp(-dist_factor)) / 2
                vel_con = (1 + np.exp(-0.5 * np.tanh(np.linalg.norm(self.obs["observation"][3:])))) / 2
                constraint1 = np.maximum(constraint1, vel_con * dist_con)
                constraint2 = np.maximum(np.linalg.norm(self.env.goal - self.obs["achieved_goal"]) - init_dist,
                                         constraint2)





        return Objective/self.T,constraint1,-constraint2/init_dist,state


    def reset(self,x0=None):
        self.obs = self.env.reset()
        self.Fail=False
        self.at_boundary=False

        if x0 is not None:
            x=x0.copy()
            Kp=500*np.eye(3)
            Kd=np.sqrt(Kp)*0.1
            x[:3]+=self.env.goal.squeeze()
            for i in range(5000):
                bias = self.ID.g()
                M = self.ID.M()

                J = self.ID.Jp(self.id)

                Mx, Mx_inv = self.ID.Mx(M, J)
                wM_des = np.dot(Kp, (x[:3] - self.obs["achieved_goal"])) - np.dot(Kd, self.obs["observation"][3:]-x[3:])
                u = -bias
                u += np.dot(J.T, np.dot(Mx, wM_des))
                self.obs, reward, done, info = self.env.step(u)

            #print(self.obs["observation"]-x)




    def set_params(self, params):
        q1 = np.repeat(np.power(10, params[0]),3)
        q2 = np.sqrt(q1)*0.1
        updated_params = np.hstack((q1.squeeze(), q2.squeeze()))
        return updated_params


class GoSafe_Optimizer(object):

    def __init__(self, low1=0.85, low2=-0.05, lengthscale=0.8, ARD=True, variance=0.1**2):
        self.low = [low1, low2]
        q =0.5
        r = 0.01
        self.params = np.asarray([q,r])

        self.rollout_limit = 500
        self.sys = System(rollout_limit=self.rollout_limit)
        f, g1, g2,state = self.sys.simulate(self.params,update=True)
        print(f, g1, g2)
        f = np.asarray([[f]])
        f = f.reshape(-1, 1)
        g1 = np.asarray([[g1]])
        g1 = g1.reshape(-1, 1)
        g2 = np.asarray([[g2]])
        g2 = g2.reshape(-1, 1)
        x0=state[0][:,2:]

        x=np.asarray(state).squeeze()
        L=[lengthscale*0.5,lengthscale*0.5,1.5*lengthscale,1.5*lengthscale,1.5*lengthscale,10*lengthscale,10*lengthscale,10*lengthscale]
        KERNEL_f = GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=variance)
        KERNEL_g = GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=variance)
        gp1 = GPy.models.GPRegression(x[0,:].reshape(1,-1), f, noise_var=0.01 ** 2, kernel=KERNEL_f)
        gp2 = GPy.models.GPRegression(x[0,:].reshape(1,-1), g1, noise_var=0.01 ** 2, kernel=KERNEL_g)
        gp3 = GPy.models.GPRegression(x[0,:].reshape(1,-1), g2, noise_var=0.01 ** 2, kernel=KERNEL_g)



        state_1=0.5
        state_2=1
        bounds = [[0, 6], [0.01, 10],
                           [-state_1, state_1],[-state_1, state_1],[-state_1, state_1],[-state_2,state_2],[-state_2, state_2],[-state_2, state_2]]
        self.opt = GoSafeSwarm([gp1, gp2, gp3], fmin=[-np.inf, low1, low2], bounds=bounds, beta=3,x_0=x0.reshape(-1,1),eta=0.05,tol=0.0,max_S2_steps=3,max_S1_steps=8,max_S3_steps=5,eps=0.1)#,max_data_size=1000,reset_size=500)
        #self.opt.S3_x0_ratio=1
        self.opt.safety_cutoff=0.8
        y = np.array([f, g1, g2])
        y=y.squeeze()
        self.add_data(x,y)
        #for i in range(1,self.rollout_limit):
         #   self.opt.add_new_data_point(x[i].reshape(1,-1),y)

        self.simulate_data()
    def simulate_data(self):
        p = [0.4,0.3]
        d = [0.02,0.01]

        for i in range(2):
            self.params = np.asarray([p[i],d[i]])
            f, g1, g2,state = self.sys.simulate(self.params,update=True,opt=self.opt)
            print(f,g1,g2)
            y = np.array([[f], [g1], [g2]])
            y = y.squeeze()
            self.add_data(state,y)
            #self.opt.add_new_data_point(x, y)


    def optimize(self):
        param = self.opt.optimize()
        #param_a=param[:self.opt.action_dim]

        print(param, end="")
        f, g1, g2,state = self.sys.simulate(param,update=True,opt=self.opt)
        #print(f, g1, g2,self.opt.criterion)
        constraint_satisified=g1>=self.low[0] and g2>=self.low[1]
        print(self.opt.criterion,constraint_satisified)
        y = np.array([[f], [g1], [g2]])
        y = y.squeeze()
        if not self.sys.at_boundary:
            self.add_data(state,y)
        else:
            self.opt.add_boundary_points(param)


    def add_data(self,state,y):
        idx=np.random.randint(low=0,high=500,size=100)
        for i in idx:
            x=state[i]
            self.opt.add_new_data_point(x.reshape(1, -1), y)


#opt=SafeOpt_Optimizer()
opt=GoSafe_Optimizer()
for i in range(100):
    opt.optimize()


max,f=opt.opt.get_maximum()
#max=[2.00179108e+00,4.13625539e+00, 3.34599393e+00, 7.41304209e-01,2.81500345e-01, 3.13137132e-03]
print("maximum",max)

f,g1,g2,state=opt.sys.simulate(max,update=True,render=True)





