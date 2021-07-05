
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


def train(env, agent, model_type, args):

    # linear velocity in the train process

    vel = 3

    H = args.history_window
    
    if args.load:

        agent.load_model(os.path.join(os.getcwd(), 'example', 'savefile', model_type))

    recorder = Rewardrecorder()
    
    recent_mission_results = []
        
    for eps in range(args.max_train_eps):
                
        done = False

        x, target = env.reset()

        steps_ep=0

        x = np.tile(x, (1, H))

        if agent.n_step>1:

            agent.buffer.memory = []

        while not env.t_max_reach and not done:

            steer = agent.get_action(x, True)

            u = np.array([vel, env.car.u_max[1]*steer[0][0]]).reshape([-1, 1])

            xn, r, done = env.step(u)

            xn = np.concatenate([x[:, 9:], xn], axis=1)

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

    # different linear velocity from the train process

    vel = 1.5
    
    H = args.history_window

    agent.load_model(os.path.join(os.getcwd(), 'example', 'savefile', model_type))

    recent_mission_results = []
        
    for eps in range(args.max_infer_eps):
        
        done = False

        x, target = env.reset()

        x = np.tile(x, (1, H))

        steps_ep=0

        if agent.n_step>1:

            agent.buffer.memory = []

        while not env.t_max_reach and not done:

            steer = agent.get_action(x, False)

            u = np.array([vel, env.car.u_max[1]*steer[0][0]]).reshape([-1, 1])

            xn, r, done = env.step(u)

            xn = np.concatenate([x[:, 9:], xn], axis=1)

            mask = 0 if done else 1

            if args.render:
                
                env.render()

            x = xn

            steps_ep += 1
        
        recent_mission_results.append(float(env.reach))

        if len(recent_mission_results)>10:

            recent_mission_results.pop(0)

        mission_results = 'success!' if env.reach else 'fail'
        print('{} episode | live steps : {:.2f} | '.format(eps + 1, steps_ep) + mission_results)

