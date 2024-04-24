import pickle
import numpy
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
from scipy.stats import norm, multivariate_normal



class KalmanFilter():
    def __init__(self, x_dim, z_dim):

        self.Q = np.eye(x_dim)
        self.R = np.eye(z_dim)
        self.B = None
        self.P = np.eye(x_dim)
        self.A = np.eye(x_dim)
        self.H = np.zeros((z_dim,x_dim))

        self.x = np.zeros((x_dim,1))
        self.y = np.zeros((z_dim,1))

        self.K = np.zeros((x_dim, z_dim))
        self.S = np.zeros((z_dim, z_dim))

        self._I = np.eye(x_dim)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.SI = np.zeros((z_dim, z_dim))
        self.inv = np.linalg.inv

    def predict(self, u=None, B=None, A=None, Q=None):

        if B is None:
            B = self.B
        if A is None:
            A = self.A
        if Q is None:
            Q = self.Q


        if B is not None and u is not None:
            self.x = np.dot(A, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(A, self.x)


        self.P = np.dot(np.dot(A, self.P), A.T) + Q


        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()


    def correction(self, z, R=None, H=None):

        if R is None:
            R = self.R

        if H is None:
            H = self.H

        self.y = z - np.dot(H, self.x)

        PHT = np.dot(self.P, H.T)

        self.S = np.dot(H, PHT) + R
        self.SI = self.inv(self.S)

        self.K = np.dot(PHT, self.SI)

        self.x = self.x + np.dot(self.K, self.y)

        I_KH = self._I - np.dot(self.K, H)

        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) +\
        np.dot(np.dot(self.K, R), self.K.T)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()



class Extended_KalmanFilter():
    def __init__(self, x_dim, z_dim):

        self.Q = np.eye(x_dim)
        self.R = np.eye(z_dim)
        self.B = None
        self.P = np.eye(x_dim)
        self.JA = None
        self.JH = None

        self.F = (lambda x:x)
        self.H = (lambda x:np.zeros(z_dim,1))

        self.x = np.zeros((x_dim,1))
        self.y = np.zeros((z_dim,1))

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

    def pred(self, T, dt=0.1):

        x = self.x.copy()
        X = [x]

        for i in range(int(T/dt)):
            x = self.F(x)

            X.append(x)

        return np.array(X)


    def correction(self, z, JH = None, H=None, R=None):

        if JH is None:
            if self.JH is None:
                JH_ = np.zeros((self.x_dim,self.z_dim))
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
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.likelihood = multivariate_normal.pdf(self.y, np.zeros_like(self.y), self.S)

class IMM_filter():
    def __init__(self, filters, mu, M):

        self.mu = mu
        self.M = M
        self.filters = filters
        self.N = len(filters)

        n_cand = [len(f.x) for f in filters]
        target_filter = np.argmax(n_cand)

        self.x = np.zeros(filters[target_filter].x.shape)
        self.P = np.zeros(filters[target_filter].P.shape)


        self.omega = 1/self.N*np.ones((self.N, self.N))
        self.mM = np.dot(self.mu, self.M)
        self.likelihood = np.zeros(self.N)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


    def mixing(self):

        self.xs, self.Ps = [], []
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):

            x = np.zeros(f.x.shape)

            for kf, wj in zip(self.filters, w):
                x[0:5] += kf.x[0:5] * wj

                if len(kf.x)>5 and len(x)>5:
                    x[5] = kf.x[5]

            self.xs.append(x)

            P = np.zeros(f.P.shape)

            for kf, wj in zip(self.filters, w):
                y = kf.x[0:5] - x[0:5]
                P[0:5,0:5] += wj * (np.outer(y, y) + kf.P[0:5,0:5])
                if len(kf.x)>5 and len(P)>5:
                    P[5,5] = kf.P[5,5]

            self.Ps.append(P)


    def prediction(self, mixing=True):

        if mixing:
            self.mixing()

        for i, f in enumerate(self.filters):
            if mixing:
                f.x = self.xs[i].copy()
                f.P = self.Ps[i].copy()

            f.predict()


    def predict(self, T, dt=0.1):

        X = [self.x]

        omega = self.omega.copy()
        mu = self.mu.copy()
        mM = self.mM.copy()
        likelihood = np.array([1.0, 1.0])
        filters = [copy.deepcopy(self.filters[i]) for i in range(len(self.filters))]

        for i in range(int(T/dt)):
            xs, Ps = [], []
            for j, (f, w) in enumerate(zip(filters, omega.T)):

                x = np.zeros(f.x.shape)
                for kf, wj in zip(filters, w):
                    x[0:5] += kf.x[0:5] * wj
                    if len(kf.x)>5 and len(x)>5:
                        x[5] = kf.x[5]
                xs.append(x)

                P = np.zeros(f.P.shape)
                for kf, wj in zip(filters, w):
                    y = kf.x[0:5] - x[0:5]
                    P[0:5,0:5] += wj * (np.outer(y, y) + kf.P[0:5,0:5])
                    if len(kf.x)>5 and len(P)>5:
                        P[5,5] = kf.P[5,5]
                Ps.append(P)

            for j, f in enumerate(filters):
                f.x = xs[j].copy()
                f.P = Ps[j].copy()
                f.predict()

                likelihood[j] = (f.P[0,0] + f.P[1,1])
            y = np.zeros(self.x.shape)

            mu = mM * likelihood

            mu /= np.sum(mu)
            mM = np.dot(mu, self.M)

            for i in range(self.N):
                for j in range(self.N):
                    omega[i, j] = (self.M[i, j]*mu[i]) / mM[j]


            for f, m_ in zip(filters, mu):
                y[0:5] += f.x[0:5] * m_
                if len(f.x)>5 and len(y)>5:
                    y[5]=f.x[5]
            X.append(y)

        return np.array(X)


    def merging(self, z):

        for i, f in enumerate(self.filters):
            f.correction(z)
            self.likelihood[i] = f.likelihood

        self.mu = self.mM * self.likelihood

        self.mu /= np.sum(self.mu)
        self.mM = np.dot(self.mu, self.M)


        for i in range(self.N):
            for j in range(self.N):
                self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.mM[j]


        self.x.fill(0)
        for f, mu in zip(self.filters, self.mu):
            self.x[0:5] += f.x[0:5] * mu

            if len(f.x)>5:
                self.x[5]=f.x[5]

        self.P.fill(0)
        for f, mu in zip(self.filters, self.mu):
            y = f.x[0:5] - self.x[0:5]

            self.P[0:5,0:5] += mu * (np.outer(y, y) + f.P[0:5,0:5])

            if len(f.x)>5:
                self.P[5,5] = (f.x[5]-self.x[5])**2+f.P[5,5]





class CA():
    def __init__(self, dt=0.1):
        self.dt = dt

    def step(self,x):

        dt = self.dt
        x_new = [x[0]+(x[2]+1/2*x[3]*dt)*np.cos(x[4])*dt,
                  x[1]+(x[2]+1/2*x[3]*dt)*np.sin(x[4])*dt,
                  x[2]+x[3]*dt,
                  x[3],
                  x[4]]

        return np.array(x_new)

    def H(self,x):

        return np.array([x[0],x[1],x[2],x[4]])

    def JA(self, x,dt = 0.1):

        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]

        JA_ = [[1, 0 , np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2,-(v+1/2*a*dt)*np.sin(yaw)*dt ],
                [0, 1 , np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2, (v+1/2*a*dt)*np.cos(yaw)*dt],
                [0,0,1,dt,0],
                [0,0,0,1,0],
                [0,0,0,0,1]]

        return np.array(JA_)

    def JH(self, x, dt = 0.1):
        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]


        JH_ = np.array([[1,0,0, 0,0 ],
                [0, 1, 0,0, 0],
                [0,0,1,0,0],
                [0,0,0,0,1]])

        return JH_

class CTRA():
    def __init__(self, dt=0.1):

        """
        x : [x, y, v, a, theta, theta_rate]
        """
        self.dt = dt

    def step(self, x):

        if np.abs(x[5])>0.1:
            x_new = [x[0]+x[2]/x[5]*(np.sin(x[4]+x[5]*self.dt)-
                                                     np.sin(x[4]))+
                      x[2]/(x[5]**2)*(np.cos(x[4]+x[5]*self.dt)+
                                                self.dt*x[5]*np.sin(x[4]+x[5]*self.dt)-
                                                np.cos(x[4])),
                      x[1]+x[2]/x[5]*(-np.cos(x[4]+x[5]*self.dt)+
                                                     np.cos(x[4]))+
                      x[2]/(x[5]**2)*(np.sin(x[4]+x[5]*self.dt)-
                                                self.dt*x[5]*np.cos(x[4]+x[5]*self.dt)-
                                                np.sin(x[4])),
                      x[2]+x[3]*self.dt,
                      x[3],
                      x[4]+x[5]*self.dt,
                      x[5]]

        else:
            x_new = [x[0]+x[2]*np.cos(x[4])*self.dt,
                      x[1]+x[2]*np.sin(x[4])*self.dt,
                      x[2]+x[3]*self.dt,
                      x[3],
                      x[4],
                      x[5]]

        return np.array(x_new)

    def H(self,x):

        return np.array([x[0],x[1],x[2],x[4]])

    def JA(self,x,dt = 0.1):

        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]
        r = x[5]


        # upper
        if np.abs(r)>0.1:
            JA_ = [[1,0,(np.sin(yaw+r*dt)-np.sin(yaw))/r,(-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt)-v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))+
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                    [0,1,(-np.cos(yaw+r*dt)+np.cos(yaw))/r,(-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw))+
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                    [0,0,1,dt,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,dt],
                    [0,0,0,0,0,1]]
        else:
            JA_ = [[1, 0 , np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2,-(v+1/2*a*dt)*np.sin(yaw)*dt ,0],
                    [0, 1 , np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2, (v+1/2*a*dt)*np.cos(yaw)*dt,0],
                    [0,0,1,dt,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,dt],
                    [0,0,0,0,0,1]]

        return np.array(JA_)

    def JH(self,x, dt = 0.1):
        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]
        r = x[5]

        # upper
        if np.abs(r)>0.1:

            JH_ = [[1,0,(np.sin(yaw+r*dt)-np.sin(yaw))/r,(-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt)-v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))+
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                    [0,1,(-np.cos(yaw+r*dt)+np.cos(yaw))/r,(-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw))+
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                    [0,0,1,dt,0,0],
                    [0,0,0,0,1,dt]]

        else:
            JH_ = [[1, 0 , np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2,-(v+1/2*a*dt)*np.sin(yaw)*dt ,0],
                    [0, 1 , np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2, (v+1/2*a*dt)*np.cos(yaw)*dt,0],
                    [0,0,1,dt,0,0],
                    [0,0,0,0,1,dt]]

        return np.array(JH_)

    def pred(self, x, t_pred):
        self.x = x

        x_list = [self.x]
        for t in range(int(t_pred/self.dt)):
            x_list.append(self.step(self.x))

        return np.array(x_list)


def simulation(sample, rate):

    """
    plot area
    """
    clip_val = np.array([[-40,20],[-15,15]])

    """
    Data parsing
    """
    pose = sample['pose']
    vel = sample['vel']
    theta = sample['heading']
    map_coords = sample['map']

    """
    a와 yaw rate은 v와 heading의 변화량 (비교용)
    """

    a_cand = (vel[1:]-vel[0:-1])*10
    a = np.insert(a_cand, 0, a_cand[0], axis=0)

    theta_rate_cand = (theta[1:]-theta[0:-1])*10
    theta_rate = np.insert(theta_rate_cand, 0, theta_rate_cand[0], axis=0)


    """
    IMM Filtering
    """
    num_of_model = 2
    mat_trans = np.array([[0.85,0.15], [0.15,0.85]])
    mu = [1.0,0.0]

    filters = [Extended_KalmanFilter(5,4),Extended_KalmanFilter(6,4)]
    models = [CA(), CTRA()]

    Q_list = [[0.1,0.1,0.1,0.1,0.001],
          [0.1,0.1,0.1,0.1,0.001,0.01]]

    x = [np.array([pose[0,0], pose[0,1], vel[0], 0, theta[0]]),
              np.array([pose[0,0], pose[0,1], vel[0], 0, theta[0],0])]

    for i in range(len(filters)):
        filters[i].F=models[i].step
        filters[i].H=models[i].H
        filters[i].JA=models[i].JA
        filters[i].JH=models[i].JH
        filters[i].Q = np.diag(Q_list[i])
        filters[i].R = np.diag([0.1,0.1,0.1,0.1])
        filters[i].x = x[i]


    IMM = IMM_filter(filters, mu, mat_trans)
    MM = [mu]
    X = [x[1]]
    Traj = []

    fig = plt.figure(0, figsize=(8,8))


    for i in range(len(pose)):

        z= [pose[i,0], pose[i,1], vel[i], theta[i]]

        IMM.prediction()
        IMM.merging(z)

        traj = IMM.predict(1)

        Traj.append(traj)
        MM.append(IMM.mu.copy())
        X.append(IMM.x.copy())

        """
        Plot map data
        """
        for k in range(len(map_coords)):
            plt.plot(map_coords[k,:,0], map_coords[k,:,1],color='k', alpha=0.4, linewidth=1.4)

        """
        Plot True trajectory
        """
        plt.plot(pose[:,0],pose[:,1],'ro--', markersize=10, alpha=0.4)

        """
        Plot Kalman filtered trajectory
        """
        plt.plot(traj[:,0],traj[:,1],'bv--',markersize=8, alpha=0.3)

        """
        Plot each model's trajectory
        """
        j=0
        temp =IMM.filters[j].pred(1)
        plt.plot(temp[:,0],temp[:,1],'s', color='g',markersize=6, alpha=0.3)

        j=1
        temp =IMM.filters[j].pred(1)
        plt.plot(temp[:,0],temp[:,1],'s', color='c',markersize=6, alpha=0.3)

        plt.xlim(clip_val[0])
        plt.ylim(clip_val[1])


        plt.pause(rate)
        plt.cla()

    plt.show()

    MM = np.array(MM)
    plt.figure(figsize=(10,10))
    for i in range(2):
        plt.plot(MM[:,i])
    plt.legend(['ca','ctra'])
    plt.show()


def main():

    rate  = 0.1

    try:
        rate = float(sys.argv[1]) # plot speed
    except:
        pass

    print('IMM start')

    with open("./sample_imm/imm.pickle", 'rb') as f:
        sample = pickle.load(f)

    simulation(sample, rate)




if __name__ == '__main__':
    main()
