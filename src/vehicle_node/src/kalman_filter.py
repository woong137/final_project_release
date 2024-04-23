import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.stats import norm, multivariate_normal


class Extended_KalmanFilter():
    def __init__(self, x_dim, z_dim):

        self.Q = np.eye(x_dim)
        self.R = np.eye(z_dim)
        self.B = None
        self.P = np.eye(x_dim)
        self.JA = None
        self.JH = None

        self.F = (lambda x: x)
        self.H = (lambda x: np.zeros(z_dim, 1))

        self.x = np.zeros((x_dim, 1))
        self.y = np.zeros((z_dim, 1))

        self.K = np.zeros((x_dim, z_dim))
        self.S = np.zeros((z_dim, z_dim))

        self.x_dim = x_dim
        self.z_dim = z_dim

        self._I = np.eye(x_dim)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.SI = np.zeros((z_dim, z_dim))
        self.inv = np.linalg.inv

        self.likelihood = 1.0

    def predict(self, u=None, JA=None, F=None, Q=None):

        if Q is None:
            Q = self.Q

        # x = Fx + Bu
        if JA is None:
            if self.JA is None:
                JA_ = np.eye(self.x_dim)
            else:
                JA_ = self.JA(self.x)
        else:
            JA_ = JA(self.x)

        if F is None:
            F = self.F

        self.x = F(self.x)

        # P = FPF' + Q
        self.P = np.dot(np.dot(JA_, self.P), JA_.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def correction(self, z, JH=None, H=None, R=None):

        if JH is None:
            if self.JH is None:
                JH_ = np.zeros((self.x_dim, self.z_dim))
            else:
                JH_ = self.JH(self.x)
        else:
            JH_ = JH(self.x)

        if H is None:
            H = self.H

        z_pred = H(self.x)

        if R is None:
            R = self.R

        self.y = z - z_pred

        PHT = np.dot(self.P, JH_.T)

        self.S = np.dot(JH_, PHT) + R
        self.SI = self.inv(self.S)

        self.K = np.dot(PHT, self.SI)

        self.x = self.x + np.dot(self.K, self.y)

        I_KH = self._I - np.dot(self.K, JH_)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + \
            np.dot(np.dot(self.K, R), self.K.T)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.likelihood = multivariate_normal.pdf(
            self.y, np.zeros_like(self.y), self.S)

class CTRA():
    def __init__(self, dt=0.1):
        """
        x : [x, y, v, a, theta, theta_rate]
        """
        self.dt = dt

    def step(self, x):

        if np.abs(x[5]) > 0.1:
            self.x = [x[0]+x[2]/x[5]*(np.sin(x[4]+x[5]*self.dt) -
                                      np.sin(x[4])) +
                      x[2]/(x[5]**2)*(np.cos(x[4]+x[5]*self.dt) +
                                      self.dt*x[5]*np.sin(x[4]+x[5]*self.dt) -
                                      np.cos(x[4])),
                      x[1]+x[2]/x[5]*(-np.cos(x[4]+x[5]*self.dt) +
                                      np.cos(x[4])) +
                      x[2]/(x[5]**2)*(np.sin(x[4]+x[5]*self.dt) -
                                      self.dt*x[5]*np.cos(x[4]+x[5]*self.dt) -
                                      np.sin(x[4])),
                      x[2]+x[3]*self.dt,
                      x[3],
                      x[4]+x[5]*self.dt,
                      x[5]]

        else:
            self.x = [x[0]+x[2]*np.cos(x[4])*self.dt,
                      x[1]+x[2]*np.sin(x[4])*self.dt,
                      x[2]+x[3]*self.dt,
                      x[3],
                      x[4],
                      x[5]]

        return self.x

    def H(self, x):

        return np.array([x[0], x[1], x[2], x[4]])

    def JA(self, x, dt=0.1):

        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]
        r = x[5]

        # upper
        if np.abs(r) > 0.1:
            JA_ = [[1, 0, (np.sin(yaw+r*dt)-np.sin(yaw))/r, (-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt) -
                     v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw)) +
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                   [0, 1, (-np.cos(yaw+r*dt)+np.cos(yaw))/r, (-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt) -
                     v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw)) +
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                   [0, 0, 1, dt, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, dt],
                   [0, 0, 0, 0, 0, 1]]
        else:
            JA_ = [[1, 0, np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2, -(v+1/2*a*dt)*np.sin(yaw)*dt, 0],
                   [0, 1, np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2,
                    (v+1/2*a*dt)*np.cos(yaw)*dt, 0],
                   [0, 0, 1, dt, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, dt],
                   [0, 0, 0, 0, 0, 1]]

        return np.array(JA_)

    def JH(self, x, dt=0.1):
        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]
        r = x[5]

        # upper
        if np.abs(r) > 0.1:

            JH_ = [[1, 0, (np.sin(yaw+r*dt)-np.sin(yaw))/r, (-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt) -
                     v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw)) +
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                   [0, 1, (-np.cos(yaw+r*dt)+np.cos(yaw))/r, (-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt) -
                     v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw)) +
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                   [0, 0, 1, dt, 0, 0],
                   [0, 0, 0, 0, 1, dt]]

        else:
            JH_ = [[1, 0, np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2, -(v+1/2*a*dt)*np.sin(yaw)*dt, 0],
                   [0, 1, np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2,
                    (v+1/2*a*dt)*np.cos(yaw)*dt, 0],
                   [0, 0, 1, dt, 0, 0],
                   [0, 0, 0, 0, 1, dt]]

        return np.array(JH_)

    def pred(self, x, t_pred):
        self.x = x

        x_list = [self.x]
        for t in range(int(t_pred/self.dt)):
            x_list.append(self.step(self.x))

        return np.array(x_list)


def simulation(sample):
    """
    Data parsing
    """
    pose = sample['pose']
    vel = sample['vel']
    theta = sample['heading']

    """
    a와 yaw rate은 v와 heading의 변화량 (비교용)
    """

    a_cand = (vel[1:]-vel[0:-1])*10
    a = np.insert(a_cand, 0, a_cand[0], axis=0)

    theta_rate_cand = (theta[1:]-theta[0:-1])*10
    theta_rate = np.insert(theta_rate_cand, 0, theta_rate_cand[0], axis=0)

    """
    Kalman Filter initialize
    """
    x_init = [pose[0, 0], pose[0, 1], vel[0], 0, theta[0], 0]
    model = CTRA(0.1)

    kf = Extended_KalmanFilter(6, 4)

    kf.F = model.step
    kf.JA = model.JA
    kf.H = model.H
    kf.JH = model.JH

    kf.x = x_init

    X = [x_init]

    for i in range(len(pose)):
        # x = [x, y, v, a, theta, theta_rate]
        # z = [x, y, v, theta]
        x = [pose[i, 0], pose[i, 1], vel[i], a[i], theta[i], theta_rate[i]]
        z = [pose[i, 0], pose[i, 1], vel[i], theta[i]]

        kf.predict(Q=np.diag([1, 1, 1, 10, 10, 100]))
        kf.correction(z=z, R=np.diag([1, 1, 1, 1]))

        model_kf = copy.deepcopy(model)

        XX = model_kf.pred(kf.x, t_pred=1)
        YY = model.pred(x, t_pred=1)

        X.append(kf.x)