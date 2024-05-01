import random
from collections import deque

class ReplayBuffer:
    def __init__(self):
        self.buffer = []
    
    def add(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))
    
    def sample(self, batch_size): #to sample from the buffer
        sample_size = min(len(self.buffer), batch_size)
        samples = random.sample(self.buffer, sample_size)
        states, actions, rewards, dones = zip(*samples)
        return states, actions, rewards, dones
    
    def __len__(self):
        return len(self.buffer)
