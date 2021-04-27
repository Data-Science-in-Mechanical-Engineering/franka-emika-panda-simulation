import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.robotics import utils

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE=500

INITIAL_q={'panda_joint1':0,'panda_joint2':0,'panda_joint3':0, 'panda_joint4':0,'panda_joint5':0,
      'panda_joint6':0,'panda_joint7':0,'panda_finger_joint1':0,'panda_finger_joint2':0}
class PandaEnv(gym.GoalEnv):
    """
    n_substeps: : int
        Optional number of MuJoCo steps to run for every call to :meth:`.step`.
        Buffers will be swapped only once per step.
    """

    def __init__(self,n_substeps=1,initial_qpos=INITIAL_q,n_actions=9):

        self.model=mujoco_py.load_model_from_path("assets/Panda_xml/model_torque.xml")
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

        self.goal = self._sample_goal()
        obs=self._get_obs()

        self.torque_ranges=np.ones(n_actions)*87
        self.torque_ranges[4:7]=12
        self.torque_ranges[7:9]=70
        self.action_space=spaces.Box(-self.torque_ranges,self.torque_ranges,dtype='float32')
        self.n_actions=n_actions
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))




    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(PandaEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

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
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('panda:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric


        achieved_goal = grip_pos.copy()

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
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
        return True

    def _sample_goal(self):
        scale=np.eye(3)
        scale[0,0]=3.5
        scale[1,1]=3.5
        scale[2, 2] = 0.35
        goal = np.dot(scale,self.initial_gripper_xpos[:3])
        goal[0]=goal[0]+ self.np_random.uniform(-self.target_range, self.target_range, size=1)
        goal[1] = goal[1] + self.np_random.uniform(-self.target_range, self.target_range, size=1)
        return goal.copy()

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal-sites_offset[site_id]
        self.sim.forward()

    def compute_reward(self, achieved_goal, goal, info):

        return 0










