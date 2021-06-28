import sys
import os
import ray
import datetime
import time
import numpy as np
import math
import argparse

from rl_agent.sac import REDQAgent
from rl_agent.utils import Rewardrecorder
from vehicle_env.navi_maze_env_car import NAVI_ENV

def train(env, agent, model_type, args):

    if args.load:

        agent.load_model(os.path.join(os.getcwd(), 'example', 'savefile', model_type))

    recorder = Rewardrecorder()
    
    recent_mission_results = []
        
    for eps in range(args.max_train_eps):
        
        done = False

        x, target = env.reset()

        steps_ep=0

        if agent.n_step>1:

            agent.buffer.memory = []

        while not env.t_max_reach and not done:

            steer = agent.get_action(x, True)

            u = np.array([1.5, env.car.u_max[1]*steer[0][0]]).reshape([-1, 1])

            xn, r, done = env.step(u)

            mask = 0 if done else 1

            if args.render:
                
                env.render()

            agent.push_samples(x, steer, r, xn, mask)

            agent.train_model()

            x = xn

            steps_ep += 1
        
        recent_mission_results.append(float(env.reach))

        if len(recent_mission_results)>10:

            recent_mission_results.pop(0)

        mission_results = 'success!' if env.reach else 'fail'
        progress_status = 'train...' if agent.sample_enough else 'explore'

        recorder.push((steps_ep, float(agent.sample_enough)))

        print('{} episode | live steps : {:.2f} | '.format(eps + 1, steps_ep) + mission_results + " | " + progress_status)

        if np.mean(recent_mission_results) > 0.99:

            print("save...")

            recorder.save(os.path.join(os.getcwd(), 'example', 'savefile', model_type))

            agent.save_model(os.path.join(os.getcwd(), 'example', 'savefile', model_type))

            break

    if np.mean(recent_mission_results) <= 0.99:

        print("end...")

        recorder.save(os.path.join(os.getcwd(), 'example', 'savefile', model_type))

        agent.save_model(os.path.join(os.getcwd(), 'example', 'savefile', model_type), False)



def infer(env, agent, model_type, args):

    agent.load_model(os.path.join(os.getcwd(), 'example', 'savefile', model_type))

    recent_mission_results = []
        
    for eps in range(args.max_infer_eps):
        
        done = False

        x, target = env.reset()

        steps_ep=0

        if agent.n_step>1:

            agent.buffer.memory = []

        while not env.t_max_reach and not done:

            steer = agent.get_action(x, False)

            u = np.array([1.5, env.car.u_max[1]*steer[0][0]]).reshape([-1, 1])

            xn, r, done = env.step(u)

            mask = 0 if done else 1

            if args.render:
                
                env.render()

            x = xn

            steps_ep += 1
        
        recent_mission_results.append(float(env.reach))

        if len(recent_mission_results)>10:

            recent_mission_results.pop(0)

        mission_results = 'success!' if env.reach else 'fail'
        progress_status = 'train...' if agent.sample_enough else 'explore'
        print('{} episode | live steps : {:.2f} | '.format(eps + 1, steps_ep) + mission_results + " | " + progress_status)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Soft actor critic algorithm with PyTorch in a 2D vehicle environment')

    parser.add_argument('--infer_only', action='store_true', default=False,
                        help='do not train the agent in the environment')

    parser.add_argument('--load', action='store_true', default=False,
                        help='copy & paste the saved model name, and load it')

    parser.add_argument('--max_train_eps', type=int, default=200,
                        help='maximum number of episodes for training (default: 200)')

    parser.add_argument('--max_infer_eps', type=int, default=5,
                        help='maximum number of episodes for inference (default: 5)')

    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment on training or inference')

    args = parser.parse_args()


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
        u_min=[0, -np.pi/6],
        u_max=[2, np.pi/6],
        reward_type='polar',
        target_fix=target,
        level=2, t_max=3000, obs_list=obs_list)

    agent = REDQAgent(
        state_size=9,
        action_size=1,
        hidden_size=64,
        buffer_size=2**14,
        minibatch_size=128,
        exploration_step=3000
    )

    model_type = 'redq'

    if not args.infer_only:

        train(env, agent, model_type, args)

    infer(env, agent, model_type, args)

