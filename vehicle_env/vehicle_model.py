import numpy as np
import math
import itertools

class UNICAR(object):
    
    def __init__(self, dT=0.05, x_init=[0.0, 0.0, 0.0], u_min=[0, -np.pi/6], u_max=[2, np.pi/6], noise_w=[0.3, 0.1]):
        
        self.x_init= np.array(x_init, dtype=np.float32).reshape([-1,1])

        self.x_dim = 3
        self.u_dim = 2
        
        self.u_min = u_min
        self.u_max = u_max

        self.noise_w = noise_w

        self.pos_min = [-20.0, -20.0]
        self.pos_max = [20.0, 20.0]
        
        self.dT = dT

    def init_state(self):

        x_state = self.x_init
        self.x = x_state
        return x_state

    def step(self, u):

        u[0, :] = np.clip(u[0, :] + self.noise_w[0] * np.random.randn(1), self.u_min[0], self.u_max[0])
        u[1, :] = np.clip(u[1, :], self.u_min[1], self.u_max[1])

        dx0 = u[0, :]*np.cos(self.x[2, :])
        dx1 = u[0, :]*np.sin(self.x[2, :])
        dx2 = u[1, :] + self.noise_w[1] * np.random.randn(1)

        dx_state = np.concatenate([dx0.reshape([-1, 1]), dx1.reshape([-1, 1]),dx2.reshape([-1, 1])], axis=0)

        self.x = self.x + self.dT*dx_state

        self.x[1,:] = np.clip(self.x[1, :], self.pos_min[1], self.pos_max[1])
        self.x[0,:] = np.clip(self.x[0, :], self.pos_min[0], self.pos_max[0])

        if self.x[2,:] < -np.pi:
            self.x[2,:] = self.x[2,:] + 2*np.pi
        
        elif self.x[2,:] > np.pi:
            self.x[2,:] = self.x[2,:] - 2*np.pi
        
        else:
            pass

        return self.x

    def calc_jacobian(self, x, u):

        F = np.zeros((self.x_dim, self.x_dim), dtype=float)
        G = np.zeros((self.x_dim, self.u_dim), dtype=float)

        F[0,2] = -u[0, :]*np.sin(x[2, :])
        F[1,2] = u[0, :]*np.cos(x[2, :])
        
        G[0,1] = np.cos(x[2, :])
        G[1,1] = np.sin(x[2, :])
        G[2,0] = 1. 

        return F, G
