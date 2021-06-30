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
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        self.fc_module=nn.Sequential(
            self.fc1,
            nn.SELU(inplace=True),
            self.fc2,
            nn.SELU(inplace=True)
        )
        
    def forward(self, x):

        x = self.fc_module(x)
        mu = self.fc3(x)
        log_std = self.fc4(x)
        
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = 2*torch.exp(log_std)

        return mu, std

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, critic_id):
        super().__init__()

        self.critic_id = critic_id

        # Q1 architecture
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.fc_module1=nn.Sequential(
            self.fc1,
            nn.SELU(inplace=True),
            self.fc2,
            nn.SELU(inplace=True)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        x1 = self.fc_module1(x)

        q_value1 = self.fc3(x1)

        return q_value1



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

class REDQAgent(object):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size,
        buffer_size=2**13,
        minibatch_size=256,
        gamma=0.99,
        n_step=3,
        tau=0.01,
        lr_a=3e-4,
        lr_c=3e-4,
        lr_alpha=1e-4,
        train_alpha=True,
        exploration_step=5000,
        N=5,
        M=2,
        G=10
        ):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.n_step = n_step
        self.tau = tau
        self.exploration_step = exploration_step

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.N = N # number of critics in the ensemble
        self.M = M # number of target critics that are randomly selected
        self.G = G # Updates per step ~ UTD-ratio

        self.actor = Actor(state_size, action_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a, eps=1e-5)

        self.critic_list = []
        self.target_critic_list = []
        self.critic_optimizer_list = []

        for i in range(self.N):

            critic = Critic(state_size, action_size, hidden_size, i).to(self.device)
            optimizer = optim.Adam(critic.parameters(), lr=lr_c, eps=1e-5)
            
            target_critic = Critic(state_size, action_size, hidden_size, i).to(self.device)
            
            target_critic.load_state_dict(critic.state_dict())

            self.critic_list.append(critic)
            self.target_critic_list.append(target_critic)
            self.critic_optimizer_list.append(optimizer)

        if self.n_step==1:
            
            self.buffer = ExperienceReplayMemory(self.buffer_size)

        else:

            self.buffer = NStepMemory(self.n_step, self.gamma, self.buffer_size)

        self.train_alpha = train_alpha

        if self.train_alpha:

            self.target_entropy = -torch.prod(torch.Tensor(action_size).to(self.device)).item()
            self.log_alpha = torch.full((1,), math.log(0.2), requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

            self.alpha = torch.exp(self.log_alpha)

        else:

            self.alpha = 0.2

        self.sample_enough = False

    def soft_target_update(self, critic, target_critic):
        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_action(self, states, is_training):
    
        mu, std = self.actor(torch.Tensor(states).to(self.device))

        if is_training:
            
            if self.sample_enough:

                normal = Normal(mu, std)

            else:

                normal = Normal(mu, 100)
        
        else:

            normal = Normal(mu, 0.001)
        
        z = normal.rsample() # reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(z)

        return action.data.cpu().numpy()

    def eval_action(self, states, epsilon=1e-6):

        mu, std = self.actor(torch.Tensor(states).to(self.device))
        normal = Normal(mu, std)
        z = normal.rsample() # reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(z)
        log_prob = normal.log_prob(z)

        # Enforcing Action Bounds
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_policy = log_prob.sum(1, keepdim=True)

        return action, log_policy

    def push_samples(self, state, action, reward, next_state, mask):
        
        self.buffer.push((state, action, reward, next_state, mask))
        
        if self.n_step==1:
            
            self.sample_enough = True if len(self.buffer.memory) > self.exploration_step else False
            
        else:

            self.sample_enough = True if len(self.buffer.main_buffer.memory) > self.exploration_step else False
            
            
    def train_model(self):

        if self.sample_enough:

            # -----------critic update

            for _ in range(self.G):

                if self.n_step==1:
                    
                    mini_batch = self.buffer.sample(self.minibatch_size)
                
                else:
                
                    mini_batch = self.buffer.main_buffer.sample(self.minibatch_size)
                
                mini_batch = np.array(mini_batch)
                states = np.vstack(mini_batch[:, 0])
                actions = list(mini_batch[:, 1])
                rewards = list(mini_batch[:, 2])
                next_states = np.vstack(mini_batch[:, 3])
                masks = list(mini_batch[:, 4])

                actions = torch.Tensor(actions).detach().to(self.device).squeeze(1)
                rewards = torch.Tensor(rewards).to(self.device)
                masks = torch.Tensor(masks).to(self.device)
            
                idx = np.random.choice(self.N, self.M, replace=False)

                next_policy, next_log_policy = self.eval_action(next_states)

                target_next_q_value1 = self.target_critic_list[idx[0]](torch.Tensor(next_states).to(self.device), next_policy)
                target_next_q_value2 = self.target_critic_list[idx[1]](torch.Tensor(next_states).to(self.device), next_policy)
                
                min_target_next_q_value = torch.min(target_next_q_value1, target_next_q_value2)
                
                if self.train_alpha:

                    min_target_next_q_value = min_target_next_q_value.squeeze(1) - self.alpha.to(self.device) * next_log_policy.squeeze(1)
                
                else:
                
                    min_target_next_q_value = min_target_next_q_value.squeeze(1) - self.alpha * next_log_policy.squeeze(1)

                if self.n_step == 1:
                
                    target = rewards + masks * self.gamma * min_target_next_q_value
                
                else:
                
                    target = rewards + masks * (self.gamma**self.n_step) * min_target_next_q_value

                for i, (critic, target_critic, optimizer) in enumerate(zip(self.critic_list, self.target_critic_list, self.critic_optimizer_list)):

                    criterion = torch.nn.MSELoss()

                    q_value = critic(torch.Tensor(states).to(self.device), actions)

                    critic_loss = criterion(q_value.squeeze(1), target.detach()) 
                
                    optimizer.zero_grad()
                    critic_loss.backward()
                    optimizer.step()

                    self.soft_target_update(critic, target_critic)

            # update actor
            policy, log_policy = self.eval_action(states)
            
            # q_value1 = self.critic_list[idx[0]](torch.Tensor(states).to(self.device), policy)
            # q_value2 = self.critic_list[idx[1]](torch.Tensor(states).to(self.device), policy)
            
            # min_q_value = torch.min(q_value1, q_value2)
            
            # if self.train_alpha:
            #     actor_loss = ((self.alpha.to(self.device) * log_policy) - min_q_value).mean() 
            # else:        
            #     actor_loss = ((self.alpha * log_policy) - min_q_value).mean() 
            

            q_mean = 0

            for i in range(self.N):

                q_value1 = self.critic_list[i](torch.Tensor(states).to(self.device), policy)

                q_mean = q_mean + q_value1

            q_mean = q_mean/self.N
            
            if self.train_alpha:
                actor_loss = ((self.alpha.to(self.device) * log_policy) - q_mean).mean() 
            else:        
                actor_loss = ((self.alpha * log_policy) - q_mean).mean() 

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.train_alpha:
                    
                # update alpha
                alpha_loss = -(self.log_alpha.to(self.device) * (log_policy + self.target_entropy).detach()).mean() 
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = torch.exp(self.log_alpha)
            
        else:

            pass
    
    def save_model(self, save_path, is_best=True):
    
        if is_best:

            for i in range(self.N):
                
                torch.save(self.critic_list[i].state_dict(), os.path.join(save_path, f"redq_critic_{i}_best.pth"))
            
            torch.save(self.actor.state_dict(), os.path.join(save_path, "redq_actor_best.pth"))

        else:

            for i in range(self.N):
                
                torch.save(self.critic_list[i].state_dict(), os.path.join(save_path, f"redq_critic_{i}_end.pth"))
            
            torch.save(self.actor.state_dict(), os.path.join(save_path, "redq_actor_end.pth"))

    def load_model(self, load_path, is_best=True):
        
        if is_best:
    
            for i in range(self.N):

                self.critic_list[i].load_state_dict(torch.load(os.path.join(load_path, f"redq_critic_{i}_best.pth")))

            self.actor.load_state_dict(torch.load(os.path.join(load_path, "redq_actor_best.pth")))
        else:
        
            for i in range(self.N):

                self.critic_list[i].load_state_dict(torch.load(os.path.join(load_path, f"redq_critic_{i}_end.pth")))

            self.actor.load_state_dict(torch.load(os.path.join(load_path, "redq_actor_end.pth")))