import sys
import os
import ray
import datetime
import time
import numpy as np
import math
import argparse
import torch
import random

from rl_agent.redq import REDQAgent
from rl_agent.utils import Rewardrecorder, infer, train
from vehicle_env.navi_maze_env_car import NAVI_ENV


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Soft actor critic algorithm with PyTorch in a 2D vehicle environment')

    parser.add_argument('--task', type=str, default='maze',
                        help='type of tasks in the environment')

    parser.add_argument('--max_infer_eps', type=int, default=5,
                        help='maximum number of episodes for inference (default: 5)')

    parser.add_argument('--history_window', type=int, default=3,
                        help='history window of observation from environment (default: 3)')

    parser.add_argument('--G', type=int, default=20,
                        help='critic gradient steps (default: 20)')

    parser.add_argument('--N', type=int, default=10,
                        help='the number of ensemble models (default: 10)')

    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment on training or inference')

    parser.add_argument('--seed', type=int, default=1,
                        help='the seed number of numpy and torch (default: 1)')

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    target = np.array([0, -16]).reshape([-1, 1])

    env = NAVI_ENV(
        dT=0.1,
        x_init=[-16.0, 16.0, 0],
        u_min=[0, -np.pi/4],
        u_max=[4, np.pi/4],
        reward_type='polar',
        target_fix=target,
        level=2, t_max=3000, obs_list=obs_list)

    agent = REDQAgent(
        state_size=env.sensor.n_sensor*args.history_window,
        action_size=1,
        hidden_size=64,
        buffer_size=2**14,
        minibatch_size=256,
        exploration_step=3000,
        N=args.N,
        G=args.G
    )

    model_type = 'redq'

    infer(env, agent, model_type, args)
