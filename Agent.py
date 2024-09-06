import torch
from torch import nn
from torch.optim import Adam
import random
from collections import deque
import numpy as np
import gymnasium as gym
from BlackjackEnv import BlackjackEnv
from QNet import QNet

class Agent():
    def __init__(self, obs_dim, act_dim, device='cpu', seed=None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.q_net = QNet(obs_dim, act_dim).to(torch.device(device))
        self.q_target = QNet(obs_dim, act_dim).to(torch.device(device))
        self.optim = Adam(self.q_net.parameters(), lr=0.0003)

        self.buffer = ReplayBuffer(seed=seed)
        self.total_steps = 0
    
    def log(self, obs, act, rew, next_obs, done):
        self.buffer.append(obs, act, rew, next_obs, done)
    
    def act(self, obs, mask, eps=0.0):
        self.q_net.eval()
        with torch.no_grad():
            action_values: torch.Tensor = self.q_net(obs)
        self.q_net.train()

        action_values = action_values.cpu().data.numpy()
        action_values[~mask] = -np.inf
        random_action = np.arange(5)

        if random.random() < eps:
            return random.choice(random_action[mask])      
        
        return np.argmax(action_values)

    def learn(self, exp, gamma):
        obs, acts, rews, next_obs, dones = exp
        q_target_next = self.q_target(next_obs).detach().max(1)[0].unsqueeze(1)
        q_targets = rews + gamma * q_target_next * (1 - dones)
        q_expected = self.q_net(obs).gather(1, acts)
        loss = nn.functional.mse_loss(q_expected, q_targets)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def soft_update(self, tau):
        for t_p, n_p in zip(self.q_target.parameters(), self.q_net.parameters()):
            t_p.data.copy_(tau * n_p.data + (1 - tau) * t_p.data)

class ReplayBuffer():
    def __init__(self, buffer_size=1000000, batch_size=128, device='cpu', seed=None):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = torch.device(device)
        random.seed(seed)

    def append(self, obs, act, rew, next_obs, done):
        if all(v is not None for v in [obs, act, rew, next_obs, done]):
            self.buffer.append(
                {
                    'obs': obs,
                    'act': act,
                    'rew': rew,
                    'next_obs': next_obs,
                    'done': done
                }
            )
    
    def sample(self):
        exp = random.sample(self.buffer, k=self.batch_size)

        obs = torch.from_numpy(np.vstack([e['obs'] for e in exp])).float().to(self.device)
        act = torch.from_numpy(np.vstack([e['act'] for e in exp])).long().to(self.device)
        rew = torch.from_numpy(np.vstack([e['rew'] for e in exp])).float().to(self.device)
        next_obs = torch.from_numpy(np.vstack([e['next_obs'] for e in exp])).float().to(self.device)
        done = torch.from_numpy(np.vstack([e['done']for e in exp])).float().to(self.device)

        return (obs, act, rew, next_obs, done)

    def __len__(self):
        return len(self.buffer)

def obs_to_tensor(obs):
    obs_arr = np.zeros(13)
    obs_arr[obs['dealer_card']] = 1
    obs_arr = np.append(obs_arr, obs['n_cards'])
    obs_arr = np.concatenate((obs_arr, obs['cards_remaining']))
    for hand in obs['hands']:
        obs_arr = np.concatenate((obs_arr, hand))
    return torch.tensor(obs_arr).float()