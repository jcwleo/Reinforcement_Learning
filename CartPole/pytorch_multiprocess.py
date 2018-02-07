import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class PAACNetwork(nn.module):
    def __init__(self, input_size, output_size):
        super(PAACNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU()
        )
        self.actor = nn.Linear(24,output_size)
        self.critic = nn.Linear(24,1)

    def forward(state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value




