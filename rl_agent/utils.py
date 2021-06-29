
import numpy as np
import random
import os
import math


class Rewardrecorder:
    def __init__(self):
        self.memory = []

    def push(self, data):
        self.memory.append(data)

    def save(self, save_path):
        np.save(os.path.join(save_path, "reward_plot.npy"), self.memory)


def infer(env, agent, model_type, args):
    
    H = args.history_window

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

