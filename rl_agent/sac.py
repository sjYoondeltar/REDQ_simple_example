import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import math
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, log_std_min=-20, log_std_max=10):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.fc5 = nn.Linear(hidden_size, action_size)

        self.fc_module=nn.Sequential(
            self.fc1,
            nn.SELU(inplace=True),
            self.fc2,
            nn.SELU(inplace=True),
            self.fc3,
            nn.SELU(inplace=True)
        )
        
    def forward(self, x):

        x = self.fc_module(x)
        mu = self.fc4(x)
        log_std = self.fc5(x)
        
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = 2*torch.exp(log_std)

        return mu, std

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()

        # Q1 architecture
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        
        self.fc_module1=nn.Sequential(
            self.fc1,
            nn.SELU(inplace=True),
            self.fc2,
            nn.SELU(inplace=True),
            self.fc3,
            nn.SELU(inplace=True)
        )

        # Q2 architecture
        self.fc5 = nn.Linear(state_size + action_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, 1)
        
        self.fc_module2=nn.Sequential(
            self.fc5,
            nn.SELU(inplace=False),
            self.fc6,
            nn.SELU(inplace=False),
            self.fc7,
            nn.SELU(inplace=False)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        x1 = self.fc_module1(x)

        q_value1 = self.fc4(x1)

        x2 = self.fc_module2(x)

        q_value2 = self.fc8(x2.clone())

        return q_value1, q_value2



class ExperienceReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class NStepMemory:
    def __init__(self, nstep, gamma, main_buffer_size):
        self.nstep = nstep
        self.memory = []
        self.gamma = gamma
        self.main_buffer=ExperienceReplayMemory(main_buffer_size)

    def clean(self):
        self.memory = []
        self.main_buffer.memory = []
        
    def push(self, transition_list):
        self.memory.append(transition_list)
        
        if len(self.memory) >= self.nstep:
        
            R = self.calc_R()
            
            transition_past = self.memory.pop(0)

            self.main_buffer.push((
                    transition_past[0],
                    transition_past[1],
                    R,
                    transition_list[3],
                    transition_list[4]
                ))

    def calc_R(self):
        R = 0.
        
        for i in reversed(range(self.nstep)):
            R = R*self.gamma + self.memory[i][2]

        return R

