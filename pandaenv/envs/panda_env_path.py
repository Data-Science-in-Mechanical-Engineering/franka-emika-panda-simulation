import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import copy
import numpy as np
from os import path
import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.robotics import utils
import scipy as sp
import scipy.interpolate
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE=500

INITIAL_q={'panda_joint1':0,'panda_joint2':0,'panda_joint3':0, 'panda_joint4':0,'panda_joint5':0,
      'panda_joint6':0,'panda_joint7':0,'panda_finger_joint1':0,'panda_finger_joint2':0}
class PandaEnvPath(gym.GoalEnv):
    """
    n_substeps: : int
        Optional number of MuJoCo steps to run for every call to :meth:`.step`.
        Buffers will be swapped only once per step.
    """

    def __init__(self,n_substeps=1,initial_qpos=INITIAL_q,n_actions=9):

        model_path = "Panda_xml/model_torque_path.xml"
        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")
        self.model=mujoco_py.load_model_from_path(fullpath)
        self.sim=mujoco_py.MjSim(self.model,nsubsteps=n_substeps)
        self.viewer=None
        self._viewers={}
        self.target_range = 0.5
        #self.metadata = {
        #    'render.modes': ['human', 'rgb_array'],
        #    'video.frames_per_second': int(np.round(1.0 / self.dt))
        #}
        self.gripper_extra_height = 0.06
        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state=copy.deepcopy(self.sim.get_state())
        self.init_pos=self.sim.data.get_site_xpos('panda:grip')
        #self.T = 0
        self.rho = 0
        self.traj = None #position targets
        self.dtraj=None  #velocity targets
        self.Tmax=1000
        self.goal = self._sample_goal()
        obs=self._get_obs()

        self.torque_ranges=np.ones(n_actions)*87
        self.torque_ranges[4:7]=12
        self.torque_ranges[7:9]=70
        self.high=self.torque_ranges
        self.low=-self.high
        self.action_space=spaces.Box(-self.torque_ranges,self.torque_ranges,dtype='float32')
        self.n_actions=n_actions
        self.rho_action=0
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        self.init_dist=1



    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.rho += self.rho_action
        self.rho = np.minimum(self.rho, 1)

        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }

        reward=-np.linalg.norm(15*obs["observation"][:3])/3 - (self.rho-1)**2*0.4-2*np.linalg.norm(action/self.action_space.high)/9
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(PandaEnvPath, self).reset()
        did_reset_sim = False
        self.traj=None
        self.dtraj=None
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        self.init_dist=np.linalg.norm(self.goal - obs["achieved_goal"])

        self.desired_target=obs["achieved_goal"]

        # sample points at random from a sphere


        self.rho=0


        traj_x=lambda rho: self.trajectory(rho,obs["achieved_goal"],self.goal,dir="x")
        dtraj_x = lambda rho: self.trajectory(rho, obs["achieved_goal"], self.goal, dir="x",velocity=True)
        traj_y = lambda rho: self.trajectory(rho, obs["achieved_goal"], self.goal, dir="y")
        dtraj_y = lambda rho: self.trajectory(rho, obs["achieved_goal"], self.goal, dir="y", velocity=True)
        traj_z = lambda rho: self.trajectory(rho, obs["achieved_goal"], self.goal, dir="z")
        dtraj_z = lambda rho: self.trajectory(rho, obs["achieved_goal"], self.goal, dir="z", velocity=True)
        #traj_x=sp.interpolate.interp1d(rho,x,kind="cubic")
        #traj_y = sp.interpolate.interp1d(rho,y, kind="cubic")
        #traj_z = sp.interpolate.interp1d(rho,z, kind="cubic")
        self.traj=[traj_x,traj_y,traj_z]
        self.dtraj=[dtraj_x,dtraj_y,dtraj_z]
        #self.T=0
        self.init_pos = self.sim.data.get_site_xpos('panda:grip')
        return obs

    def trajectory(self,rho,obs,goal,dir="x",velocity=False):
        if dir=="x":
            diff=goal[0]-obs[0]
            if not velocity:
                traj=-2*diff*rho**3 +3*diff*rho**2+obs[0]
            else:
                traj=-6*diff*rho**2 + 6*diff*rho
        elif dir=="y":
            diff = goal[1]-obs[1]
            if not velocity:
                traj = -2 * diff * rho ** 3 + 3 * diff * rho ** 2 + obs[1]
            else:
                traj = -6 * diff * rho ** 2 + 6 * diff * rho
        elif dir=="z":
            diff = goal[2] - obs[2]
            if not velocity:
                traj = -2 * diff * rho ** 3 + 3 * diff * rho ** 2 + obs[2]
            else:
                traj = -6 * diff * rho ** 2 + 6 * diff * rho

        return traj



    def close(self):
        if self.viewer is not None:
            #self.viewer.finish()
            self.viewer = None
            self._viewers = {}



    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'panda:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('panda:mocap', gripper_target)
        self.sim.data.set_mocap_quat('panda:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        self.initial_gripper_xpos = self.sim.data.get_site_xpos('panda:grip').copy()

    def _get_obs(self):
        # positions
        #grip_pos = self.sim.data.get_body_xpos('panda_rightfinger')
        #dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        #grip_velp = self.sim.data.get_body_xvelp('panda_rightfinger') * dt
        #robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        grip_pos = self.sim.data.get_site_xpos('panda:grip')
        #dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('panda:grip')#* dt
        #robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        #object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        #gripper_state = robot_qpos[-2:]
        #gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric


        achieved_goal = grip_pos.copy()

        if self.traj is not None:
            x_goal=self.traj[0](self.rho)
            y_goal=self.traj[1](self.rho)
            z_goal=self.traj[2](self.rho)
            goal=np.asarray([x_goal,y_goal,z_goal])

            v_x_goal=self.dtraj[0](self.rho)*self.rho_action
            v_y_goal = self.dtraj[1](self.rho)*self.rho_action
            v_z_goal = self.dtraj[2](self.rho)*self.rho_action
            v_goal = np.asarray([v_x_goal, v_y_goal, v_z_goal])
            #v_goal = np.zeros(3)
        else:
            goal=grip_pos.copy()
            v_goal=grip_velp
        obs=np.concatenate([grip_pos-goal,grip_velp-v_goal])


        #goal=self.goal.copy()
        #goal=goal-self.init_pos
        #goal=goal/self.Tmax*np.minimum(self.T,self.Tmax)+self.init_pos
        #self.T+=1
        self.desired_target = goal

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            #'desired_goal': self.goal.copy(),
            'desired_goal': goal.copy(),
            'velocity_EE': grip_velp,
            'velocity_goal':v_goal,
            'rho': self.rho,
        }

    def _step_callback(self):

        self.sim.data.set_joint_qpos('panda_finger_joint1', 0.)
        self.sim.data.set_joint_qpos('panda_finger_joint2', 0.)
        self.sim.forward()

    def _is_success(self, achieved_goal, desired_goal):
        dist=np.linalg.norm(achieved_goal-desired_goal,2)
        return  dist<=1e-2

    def _set_action(self,action):
        assert action.shape == (self.n_actions,)

        for i in range(self.n_actions):
            self.sim.data.ctrl[i] = action[i]
        #utils.ctrl_set_action(self.sim, action)
        #utils.mocap_set_action(self.sim, action)

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('panda_link7')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        for i in range(250):
            u = self.sim.data.qfrc_bias
            u = np.clip(u, self.action_space.low, self.action_space.high)
            self._set_action(u)
            self.sim.step()
            self._step_callback()
        return True

    def _sample_goal(self):
        scale=np.eye(3)
        scale[0,0]=4#7
        scale[1,1]=4#7
        scale[2, 2] = 0.8 #0.35
        goal = np.dot(scale,self.initial_gripper_xpos[:3])
        goal[1]+=0.25
        #goal[0]=goal[0]+ self.np_random.uniform(-self.target_range, self.target_range, size=1)
        #goal[1] = goal[1] + self.np_random.uniform(-self.target_range, self.target_range, size=1)
        return goal.copy()

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal- sites_offset[site_id]

        site_id=self.sim.model.site_name2id('path_follower')
        self.sim.model.site_pos[site_id] = self.desired_target - sites_offset[site_id]
        #additional_offset=np.zeros(3)
        #additional_offset[0]=-0.15
        #additional_offset[1]=-0.15

        #site_idtable0=self.sim.model.site_name2id('table0')
        #self.sim.model.site_pos[site_idtable0] = self.goal+additional_offset - sites_offset[site_idtable0]


        #site_idtable1 = self.sim.model.site_name2id('table1')
        #additional_offset[1] = 0.05
        #self.sim.model.site_pos[site_idtable1] = self.goal+additional_offset - sites_offset[site_idtable1]
        #self.sim.forward()











