import torch
from collections import deque
import numpy as np
from BlackjackEnv import BlackjackEnv
from Agent import *

device = torch.device('cpu')
env = BlackjackEnv()
agent = Agent(obs_dim=obs_to_tensor(env.observation_space.sample()).size(0), act_dim=env.action_space.n)

def train(n_episodes, max_t, eps_start, eps_end, eps_decay):
    eps = eps_start
    prev_rews = deque(maxlen=100)
    total_t = 0
    for i in range(n_episodes):
        total_rew = 0
        obs, info = env.reset()
        obs = obs_to_tensor(obs).to(device)

        for t in range(max_t):
            act = agent.act(obs, env.get_legal(), eps)

            next_obs, rew, term, trunc, info = env.step(act)
            next_obs = obs_to_tensor(next_obs).to(device)

            agent.log(obs, act, rew, next_obs, term)

            total_t += 1
            if total_t % 5 == 0 and len(agent.buffer) >= agent.buffer.batch_size:
                agent.learn(agent.buffer.sample(), 0.99)

            obs = next_obs
            total_rew += rew

            if term:
                break
            
        
        prev_rews.append(total_rew)
        
        eps = max(eps_end, eps * eps_decay)
        if (i + 1) % 100 == 0:
            print('avg of last 100 ep. rews: ' + str(np.mean(prev_rews)))

        if (i + 1) % 250 == 0:
            torch.save(agent.q_net.state_dict(), 'q_net_' + str(i + 1) + '.pt')
        
train(2500, 1000, 1.0, 0.01, 0.999)