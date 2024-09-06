from torch import nn

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, n_hidden=4):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_size)]
        for i in range(n_hidden - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)