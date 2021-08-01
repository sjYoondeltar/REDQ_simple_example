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

from rl_agent.sac import SACAgent
from rl_agent.utils import Rewardrecorder, Controlrecorder, infer, train
from vehicle_env.navi_maze_env_car import NAVI_ENV


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Soft actor critic algorithm with PyTorch in a 2D vehicle environment')
    
    parser.add_argument('--task', type=str, default='maze',
                        help='type of tasks in the environment')

    parser.add_argument('--device', type=str, default='cuda',
                        help='the device used for pytorch tensor between cuda and cpu (default: cuda)')

    parser.add_argument('--max_infer_eps', type=int, default=5,
                        help='maximum number of episodes for inference (default: 5)')

    parser.add_argument('--history_window', type=int, default=3,
                        help='history window of observation from environment (default: 3)')

    parser.add_argument('--G', type=int, default=1,
                        help='critic gradient update steps (default: 1)')

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

    target = np.array([0, -16]).reshape([-1, 1])

    env = NAVI_ENV(
        dT=0.1,
        x_init=[-16.0, 16.0, 0],
        u_min=[0, -np.pi/2],
        u_max=[4, np.pi/2],
        reward_type='polar',
        target_fix=target,
        level=2, t_max=3000, obs_list=obs_list)

    agent = SACAgent(
        state_size=env.sensor.n_sensor*args.history_window,
        action_size=1,
        hidden_size=64,
        buffer_size=2**14,
        minibatch_size=256,
        exploration_step=3000,
        device=args.device,
        G=args.G
    )

    model_type = 'sac' if args.G==1 else f'sac_g{args.G}'

    infer(env, agent, model_type, args)

