import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR

from random import random
import itertools

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128 ,1)
        
    def forward(self, up_price, down_price, exercise_price, cash, risk_free_factor):
        # Aggregate the three input environment value and 
        x = torch.stack([up_price, down_price, exercise_price, cash, risk_free_factor], dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# Configuration parameters
UP_FACTOR = 1.1
DOWN_FACTOR = 1 / UP_FACTOR
RISK_FREE_RATE = 0.02
RISK_FREE_FACTOR = 1 + RISK_FREE_RATE
EXERCISE_PRICE = 0.95

torch.autograd.set_detect_anomaly(True)

def variance_loss(policies, n=10):

    UP = True
    DOWN = False

    up_factor = torch.tensor([UP_FACTOR], dtype=torch.float32)
    down_factor = torch.tensor([DOWN_FACTOR], dtype=torch.float32)
    risk_free_factor = torch.tensor([RISK_FREE_FACTOR], dtype=torch.float32)
    exercise_price = torch.tensor([EXERCISE_PRICE], dtype=torch.float32)

    final_portfolio_values = []

    for transition_path in itertools.product([UP, DOWN], repeat=n):

        current_price = torch.tensor([1], dtype=torch.float32)
        cash = torch.tensor([0], dtype=torch.float32)
        holding = torch.tensor([0], dtype=torch.float32)

        for policy, up in zip(policies, transition_path):

            # Record situation the beginning of the timestep
            up_price = (current_price * up_factor).detach()
            down_price = (current_price * down_factor).detach()

            # Perform re-balancing
            hedge_ratio = policy(up_price, down_price, exercise_price, cash, risk_free_factor).view(-1)
            cash = cash - (hedge_ratio - holding) * current_price
            holding = hedge_ratio

            # The stock market moves
            if up:
                current_price = (current_price * up_factor).detach()
            else:
                current_price = (current_price * down_factor).detach()

            cash = cash * risk_free_factor.detach()

        final_portfolio_values.append(cash + holding * current_price + torch.where(exercise_price > current_price, exercise_price - current_price, torch.zeros_like(current_price)))
        
    return torch.concat(final_portfolio_values).var()

policies = [Policy() for _ in range(10)]
all_params = []
for policy in policies:
    all_params.extend(list(policy.parameters()))
                      
optimizer = optim.SGD(all_params, lr=1)
# scheduler = StepLR(optimizer, step_size=20, gamma=0.2)

chart_x = []
chart_y = []

num_epochs = 60
for epoch in range(num_epochs):

    # Zero the gradients
    optimizer.zero_grad()
    
    # Calculate the loss
    loss = variance_loss(policies)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()
    # scheduler.step()

    print(f'Epoch {epoch + 1} loss:', format(float(loss), '.8f'))
    chart_x.append(epoch + 1)
    chart_y.append(float(loss))

# Create a line chart to show convergence as training progresses
plt.plot(chart_x, chart_y)
plt.title('Deviation from Analytical Solution')
plt.xlabel('Steps')
plt.show()