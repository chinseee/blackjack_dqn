from BlackjackEnv import BlackjackEnv
from QNet import QNet
from Agent import obs_to_tensor
import torch
import numpy as np

env = BlackjackEnv()
net = QNet(79, 5)
net.load_state_dict(torch.load('q_net_2500.pt'))
net.eval()

total_rew = 0
for i in range(1000):
    obs, _ = env.reset()
    
    while True:
        legal = env.get_legal()
        net_obs = obs_to_tensor(obs)
        action_values: torch.Tensor = net(net_obs)
        action_values = action_values.cpu().data.numpy()
        action_values[~legal] = -np.inf

        obs, rew, term, _, _ = env.step(np.argmax(action_values))
        total_rew += rew
        if term:
            break

print(total_rew / 1000)