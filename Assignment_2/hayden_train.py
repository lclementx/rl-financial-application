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
optimizer = optim.SGD(policy.parameters(), lr=0.001)

def position_change(hedge_ratio, previous_price, next_price, exercise_price):
    payout = torch.where(exercise_price > next_price, exercise_price - next_price, torch.zeros_like(next_price))
    return hedge_ratio * (next_price - previous_price) - payout

def loss_function(hedge_ratio, up_price, down_price, current_price, exercise_price):
    up_position_change = position_change(hedge_ratio, current_price, up_price, exercise_price)
    down_position_change = position_change(hedge_ratio, current_price, down_price, exercise_price)
    return torch.abs(up_position_change - down_position_change).mean()

# Pre-calculate correct answer for proof of convergence
answer = analytical_answer(up_prices, down_prices, exercise_price)

# Convert exercise price into suitable
exercise_price = torch.full(previous_prices.size(), exercise_price)

chart_x = []
chart_y = []

num_epochs = 500000
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    hedge_ratio = policy(up_prices, down_prices, exercise_price).view(-1)
    
    # Calculate the loss
    loss = loss_function(hedge_ratio, up_prices, down_prices, previous_prices, exercise_price)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    if (epoch + 1) % 2000 == 0:
        print(f'Epoch {epoch + 1} loss:', format(float(loss), '.8f'))
        # Print difference from analytical answer to show convergence
        print(f'Average deviation from analytical answer {float(torch.abs(hedge_ratio - answer).mean()):.8f}')
        chart_x.append(epoch + 1)
        chart_y.append(float(torch.abs(hedge_ratio - answer).mean()))

# Save the model
torch.save(policy.state_dict(), 'hedge_policy.pt')

# Create a line chart to show convergence as training progresses
plt.plot(chart_x, chart_y)
plt.title('Deviation from Analytical Solution')
plt.xlabel('Steps')
plt.show()
