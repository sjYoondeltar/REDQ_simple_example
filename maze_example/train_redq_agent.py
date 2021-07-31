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

    parser = argparse.ArgumentParser(description='REDQ algorithm with PyTorch in a 2D vehicle environment')
    
    parser.add_argument('--task', type=str, default='maze',
                        help='type of tasks in the environment')

    parser.add_argument('--device', type=str, default='cuda',
                        help='the device used for pytorch tensor between cuda and cpu (default: cuda)')

    parser.add_argument('--load', action='store_true', default=False,
                        help='copy & paste the saved model name, and load it')

    parser.add_argument('--max_train_eps', type=int, default=200,
                        help='maximum number of episodes for training (default: 200)')

    parser.add_argument('--max_infer_eps', type=int, default=5,
                        help='maximum number of episodes for inference (default: 5)')

    parser.add_argument('--history_window', type=int, default=4,
                        help='history window of observation from environment (default: 4)')

    parser.add_argument('--G', type=int, default=20,
                        help='critic gradient update steps (default: 20)')

    parser.add_argument('--N', type=int, default=10,
                        help='the number of ensemble models (default: 10)')

    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment on training or inference')

    parser.add_argument('--version', type=str, default='v1',
                        help='REDQ version (default: v1)')

    parser.add_argument('--fix_alpha', action='store_true', default=False,
                        help='fix alpha to 0.2')

    parser.add_argument('--seed', type=int, default=1234,
                        help='the seed number of numpy and torch (default: 1234)')

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
        u_min=[0, -np.pi/4],
        u_max=[4, np.pi/4],
        reward_type='polar',
        target_fix=target,
        level=2, t_max=2000, obs_list=obs_list)

    agent = REDQAgent(
        state_size=env.sensor.n_sensor*args.history_window,
        action_size=1,
        hidden_size=64,
        buffer_size=2**14,
        minibatch_size=256,
        exploration_step=3000,
        device=args.device,
        N=args.N,
        G=args.G,
        version=args.version,
        train_alpha= not args.fix_alpha
    )

    model_type = 'redq' if args.version == 'v1' else 'redq_v2'

    if not os.path.isdir(os.path.join(os.getcwd(), 'maze_example', 'savefile', model_type)):

        os.makedirs(os.path.join(os.getcwd(), 'maze_example', 'savefile', model_type))

    train(env, agent, model_type, args)

    infer(env, agent, model_type, args)

