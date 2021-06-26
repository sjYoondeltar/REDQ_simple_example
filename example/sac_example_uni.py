import sys

import ray
import datetime
import time
import numpy as np
import math

from rl_agent.sac import SACAgent
from vehicle_env.navi_maze_env_car import NAVI_ENV

RENDER = False
TRAIN = True


if __name__ == '__main__':

    obs_list =[
        [-4.0, 8.0, 32.0, 8.0],
        [4.0, -8.0, 32.0, 8.0]
    ]
    
    obs_pts = np.array([[
        [obs[0]-obs[2]/2, obs[1]-obs[3]/2],
        [obs[0]+obs[2]/2, obs[1]-obs[3]/2],
        [obs[0]+obs[2]/2, obs[1]+obs[3]/2],
        [obs[0]-obs[2]/2, obs[1]+obs[3]/2],
        ] for obs in obs_list])

    target = np.array([16, -16]).reshape([-1, 1])

    env = NAVI_ENV(
        x_init=[-16.0, 16.0, 0],
        u_min=[0, -np.pi/6],
        u_max=[2, np.pi/6],
        reward_type='polar',
        target_fix=target,
        level=2, t_max=2000, obs_list=obs_list)

    agent = SACAgent(
        state_size=9,
        action_size=1,
        hidden_size=64
    )
        
    for eps in range(10):
        
        done = False

        x, target = env.reset()

        while not env.t_max_reach and not done:

            steer = agent.get_action(x, TRAIN)

            u = np.array([1, steer[0][0]]).reshape([-1, 1])

            xn, r, done = env.step(u)

            mask = 0 if done else 1

            if RENDER:
                
                env.render()

            agent.push_samples(x, steer, r, xn, mask)

            agent.train_model()

            x = xn
