from random import random

import torch
from torch import nn
import torch.nn.functional as F

def random_agent(Wt):
    return Wt * random()


# QNetwork
class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 1)
        self.optim = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        y = self.fc2(x)
        return y

    def estimate(self, t, W, x):
        x = torch.stack([t, W, x]).transpose()
        y = self.forward(x)
        return y.flatten().tolist()