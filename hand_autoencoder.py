import torch.nn as nn
import torch

class HandAutoEncoder(nn.Module):
    def __init__(self, init_space, hidden_dim, squeeze_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.init = nn.Linear(init_space[0], hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.squeeze = nn.Linear(hidden_dim, squeeze_dim)
        self.enlarge = nn.Linear(squeeze_dim, hidden_dim)
        self.end = nn.Linear(hidden_dim, init_space[0])
    
    def forward(self, x):
        x = self.relu(self.init(x))
        x = self.relu(self.hidden(x))
        x = self.relu(self.hidden(x))
        x = self.relu(self.squeeze(x))
        x = self.relu(self.enlarge(x))
        x = self.relu(self.hidden(x))
        x = self.relu(self.hidden(x))
        x = self.end(x)

        return x
