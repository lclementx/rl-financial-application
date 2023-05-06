import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from random import random

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128 ,1)
        
    def forward(self, up_price, down_price, exercise_price):
        # Aggregate the three input environment value and 
        x = torch.stack([up_price, down_price, exercise_price], dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

def analytical_answer(up_price, down_price, exercise_price):
    payout_up = torch.where(exercise_price > up_price, exercise_price - up_price, torch.zeros_like(exercise_price))
    payout_down = torch.where(exercise_price > down_price, exercise_price - down_price, torch.zeros_like(exercise_price))
    return (payout_up - payout_down) / (up_price - down_price)

# Configuration parameters
up_factor = 1.1
down_factor = 1 / up_factor
risk_free_rate = 0.02
exercise_price = 0.95
risk_neutral_prob = (1 + risk_free_rate - down_factor) / (up_factor - down_factor)

def generate_simulation():

    current_price = 1
    up_prices = []
    down_prices = []
    previous_prices = []
    next_prices = []
    
    for _ in range(9):
        previous_prices.append(current_price)
        up_prices.append(current_price * up_factor)
        down_prices.append(current_price * down_factor)
        if random() < risk_neutral_prob:
            current_price *= up_factor
        else:
            current_price *= down_factor
        next_prices.append(current_price)

    return up_prices, down_prices, previous_prices, next_prices

up_prices = []
down_prices = []
previous_prices = []
next_prices = []

# Sample 20 rounds of simulation
for _ in range(20):
    up, dp, pp, np = generate_simulation()
    up_prices.extend(up)
    down_prices.extend(dp)
    previous_prices.extend(pp)
    next_prices.extend(np)

up_prices = torch.tensor(up_prices)
down_prices = torch.tensor(down_prices)
exercise_price = torch.tensor(exercise_price)
previous_prices = torch.tensor(previous_prices)
next_prices = torch.tensor(next_prices)

policy = Policy()

# Pre-calculate correct answer for proof of convergence
answer = analytical_answer(up_prices, down_prices, exercise_price)

# Convert exercise price into suitable
exercise_price = torch.full(previous_prices.size(), exercise_price)

# Load the model
state_dict = torch.load('Assignment_2/hedge_policy.pt')
policy.load_state_dict(state_dict)

# Forward pass
hedge_ratio = policy(up_prices, down_prices, exercise_price).view(-1)

# Calculate deviation from analytical solution
print(f'Average deviation from analytical answer {float(torch.abs(hedge_ratio - answer).mean()):.8f}')