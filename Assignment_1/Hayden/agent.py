from random import random

import torch
from torch import nn

def random_agent(Wt, spread=5):
    return Wt * (random() - 0.5) * spread


# QNetwork
class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        y = self.fc3(x)
        return y

    def estimate(self, t, W, a):
        x = torch.tensor([t, W, a], dtype=torch.float32).transpose(0, 1)
        y = self.forward(x)
        return y.flatten().tolist()


# Policy Network
class PNetwork(nn.Module):

    def __init__(self):
        super(PNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        a = torch.tanh(self.fc2(x)) * 4
        return a

    def estimate(self, t, W):
        x = torch.tensor([t, W], dtype=torch.float32).transpose(0, 1)
        a = self.forward(x)
        return a.flatten().tolist()