import torch
from torch import nn
import utils


    
class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # TODO: Define the policy network
        self.policy = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, 2*hidden_dim),
									nn.ReLU(inplace=True),
                                    nn.Linear(2*hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, action_shape))

        self.apply(utils.weight_init)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, obs, std):
        # TODO: Define the forward pass
        mu = self.relu(self.policy(obs))
        mu = self.sig(mu)
        #print(mu)

        std = torch.ones_like(mu, dtype= float) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


