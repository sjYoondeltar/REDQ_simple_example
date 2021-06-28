import sys
import os
import ray
import datetime
import time
import numpy as np
import math

from rl_agent.sac import SACAgent
from rl_agent.utils import Rewardrecorder
from vehicle_env.navi_maze_env_car import NAVI_ENV

RENDER = False
TRAIN = True
LOAD_MODEL = False
MAX_EPISODE = 200

if __name__ == '__main__':

    G = 10

    obs_list =[
        [-12.0, 8.0, 16.0, 8.0],
        [12.0, 8.0, 16.0, 24.0],
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
        dT=0.1,
        x_init=[-16.0, 16.0, 0],
        u_min=[0, -np.pi/4],
        u_max=[2, np.pi/4],
        reward_type='polar',
        target_fix=target,
        level=2, t_max=3000, obs_list=obs_list)

    agent = SACAgent(
        state_size=9,
        action_size=1,
        hidden_size=64,
        buffer_size=2**14,
        minibatch_size=128,
        exploration_step=2000
    )

    if LOAD_MODEL:

        agent.load_model(os.path.join(os.getcwd(), 'example', 'savefile', 'sac_g10'))

    recorder = Rewardrecorder()
    
    recent_mission_results = []
        
    for eps in range(MAX_EPISODE):
        
        done = False

        x, target = env.reset()

        steps_ep=0

        if agent.n_step>1:

            agent.buffer.memory = []

        while not env.t_max_reach and not done:

            steer = agent.get_action(x, TRAIN)

            u = np.array([2, np.pi*steer[0][0]/4]).reshape([-1, 1])

            xn, r, done = env.step(u)

            mask = 0 if done else 1

            if RENDER:
                
                env.render()

            agent.push_samples(x, steer, r, xn, mask)

            if TRAIN:

                agent.train_model(G=1)

            x = xn

            steps_ep += 1
        
        recent_mission_results.append(float(env.reach))

        if len(recent_mission_results)>10:

            recent_mission_results.pop(0)

        mission_results = 'success!' if env.reach else 'fail'
        progress_status = 'train...' if agent.sample_enough else 'explore'

        recorder.push((steps_ep, float(agent.sample_enough)))

        print('{} episode | live steps : {:.2f} | '.format(eps + 1, steps_ep) + mission_results + " | " + progress_status)

        if np.mean(recent_mission_results) > 0.99 and TRAIN:

            print("save...")

            recorder.save(os.path.join(os.getcwd(), 'example', 'savefile', 'sac_g10'))

            agent.save_model(os.path.join(os.getcwd(), 'example', 'savefile', 'sac_g10'))

            break

    if np.mean(recent_mission_results) <= 0.99 and TRAIN:

        print("end...")

        recorder.save(os.path.join(os.getcwd(), 'example', 'savefile', 'sac_g10'))

        agent.save_model(os.path.join(os.getcwd(), 'example', 'savefile', 'sac_g10'), False)