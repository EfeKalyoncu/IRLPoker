import torch
import torch.nn as nn
import utils


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()


        # TODO: Define the Q network
        self.network = nn.Sequential(nn.Linear(repr_dim + action_shape, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True))	
        self.info = nn.Linear(hidden_dim, 1)	
        self.Q = nn.Linear(hidden_dim, 1)
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # Hint: Pass the state and action through the network and return the Q value
        input = torch.cat((obs.float(),action.float()), dim = 1)
        output = self.network(input)

        q =  self.Q(output)
        info = self.info(output)
        
        return q, info

