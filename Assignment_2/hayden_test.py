import torch
from torch import nn


def reward_function(hedge_ratio, previous_price, next_price, exercise_price):

    if exercise_price > next_price:
        payoff = exercise_price - next_price
    else:
        payoff = 0

    position_change = payoff + hedge_ratio * (next_price - previous_price)
    
    return torch.abs(position_change)


# Pytorch model boiler plate
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16 ,1)
        
    def forward(self, up_price, down_price, exercise_price):
        up_ratio = up_price / exercise_price
        down_ratio = down_price / exercise_price

        x = torch.stack([up_ratio, down_ratio], dim=1)
        return self.fc2(torch.tanh(self.fc1(x)))
