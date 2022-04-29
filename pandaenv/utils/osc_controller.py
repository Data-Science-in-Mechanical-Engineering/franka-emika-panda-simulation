import numpy as np
from mujoco_py import cymj

from scipy.optimize import minimize
#Inspired from https://github.com/abr/abr_control/blob/f60bf4dc47927224af610b047c85bb33e4b6d1bb/abr_control/arms/mujoco_config.py#L154
'''
Class which uses mujoco functions to set up inverse dynamics operational space controller
'''
class inverse_dynamics_control(object):

    def __init__(self,env,njoints,target=np.zeros(3),id='panda:grip',penalty=1e-5):
        self.model=env.model
        self.sim=env.sim
        self.njoints=njoints
        self.target=target
        self.id=id
        self.penalty=penalty


    def _load_state(self, q, dq=None, u=None):
        """Change the current joint angles
        Parameters
        ----------
        q: np.array
            The set of joint angles to move the arm to [rad]
        dq: np.array
            The set of joint velocities to move the arm to [rad/sec]
        u: np.array
            The set of joint forces to apply to the arm joints [Nm]
        """
        # save current state
        old_q = np.copy(self.sim.data.qpos)
        old_dq = np.copy(self.sim.data.qvel)
        old_u = np.copy(self.sim.data.ctrl)

        # update positions to specified state
        self.sim.data.qpos[:] = np.copy(q)
        if dq is not None:
            self.sim.data.qvel[:]= np.copy(dq)
        if u is not None:
            self.sim.data.ctrl[:] = np.copy(u)

        # move simulation forward to calculate new kinamtic information
        self.sim.forward()

        return old_q, old_dq, old_u

    def g(self, q=None,dq=None):
        """Returns qfrc_bias variable, which stores the effects of Coriolis,
        centrifugal, and gravitational forces
        Parameters
        ----------
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        # TODO: For the Coriolis and centrifugal functions, setting the
        # velocity before calculation is important, how best to do this?
        if q is not None:
            if dq is not None:
                old_q, old_dq, old_u = self._load_state(q,dq)
            else:
                old_q, old_dq, old_u = self._load_state(q)

        g = -1 * self.sim.data.qfrc_bias

        if q is not None:
            self._load_state(old_q, old_dq, old_u)

        return g

    def M(self,q=None,dq=None):
        _MNN_vector = np.zeros(self.njoints ** 2)
        if q is not None:
            if dq is not None:
                old_q, old_dq, old_u = self._load_state(q,dq)
            else:
                old_q, old_dq, old_u = self._load_state(q)
        cymj._mj_fullM(self.model, _MNN_vector, self.sim.data.qM) #Need qM for mass matrix
        M = _MNN_vector
        M = M.reshape([self.njoints, self.njoints])

        if q is not None:
            self._load_state(old_q, old_dq, old_u)
        return M

    def Jp(self, id, q=None,dq=None):
        """Returns the Jacobian for the specified Mujoco object
        Parameters
        ----------
        name: id
            The id of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        if q is not None:
            if dq is not None:
                old_q, old_dq, old_u = self._load_state(q, dq)
            else:
                old_q, old_dq, old_u = self._load_state(q)

        J = self.sim.data.site_jacp[id, :]
        J = J.reshape(3, -1)

        if q is not None:
            self._load_state(old_q, old_dq, old_u)

        return J


    def Jr(self, id, q=None,dq=None):
        """Returns the Jacobian for the specified Mujoco object
        Parameters
        ----------
        name: id
            The id of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        if q is not None:
            if dq is not None:
                old_q, old_dq, old_u = self._load_state(q, dq)
            else:
                old_q, old_dq, old_u = self._load_state(q)

        J = self.sim.data.site_jacr[id, :]
        J = J.reshape(3, -1)

        if q is not None:
            self._load_state(old_q, old_dq, old_u)

        return J

    def Dynamics(self,x,u):

        q=x[:self.njoints]
        dq=x[self.njoints:]

        f=np.zeros(2*self.njoints)
        f[:self.njoints]=dq

        M=self.M(q=q,dq=dq)
        bias=self.g(q=q,dq=dq)
        M_inv = np.linalg.pinv(M, rcond=1e-4)
        f[self.njoints:]=np.dot(M_inv,bias+u)



    def find_qdes(self,q_init=None):

        if q_init is None:
            q_init=self.sim.data.qpos

        res=minimize(self.IK_objective,q_init, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})

        return res.x

    def IK_objective(self,q):


        old_q, old_dq, old_u = self._load_state(q)
        w=self.sim.data.get_site_xpos(self.id)

        O=np.linalg.norm(self.target-w)**2

        O+=self.penalty*np.linalg.norm(q)**2

        self._load_state(old_q, old_dq, old_u)

        return O

    def Mx(self,M, J):
        M_inv = np.linalg.inv(M)
        Mx_inv = np.dot(J, np.dot(M_inv, J.T))
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-4)
        return Mx, Mx_inv









    #def IK_objective(self,):








