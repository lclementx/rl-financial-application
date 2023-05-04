import torch
import torch.nn as nn
import torch.optim as optim

from random import random, seed


# Pytorch model boiler plate
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32 ,1)
        
    def forward(self, up_price, down_price, exercise_price):
        up_ratio = up_price / exercise_price
        down_ratio = down_price / exercise_price

        x = torch.stack([up_ratio, down_ratio], dim=1)
        return self.fc2(torch.tanh(self.fc1(x)))

def analytical_answer(up_price, down_price, exercise_price):
    payout_up = torch.where(exercise_price > up_price, exercise_price - up_price, torch.zeros_like(exercise_price))
    payout_down = torch.where(exercise_price > down_price, exercise_price - down_price, torch.zeros_like(exercise_price))
    return (payout_up - payout_down) / (up_price - down_price)

up_factor = 1.1
down_factor = 1 / up_factor
risk_free_rate = 0.01
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
optimizer = optim.SGD(policy.parameters(), lr=0.0002)

def loss_function(hedge_ratio, previous_price, next_price, exercise_price):
    payout = torch.where(exercise_price > next_price, exercise_price - next_price, torch.zeros_like(next_price))
    position_change = hedge_ratio * (next_price - previous_price) - payout
    return torch.abs(position_change).mean()

# ans = analytical_answer(torch.tensor(1.3), torch.tensor(1 / 1.3), torch.tensor(0.9))
# uloss = loss_function(ans, torch.tensor(1), torch.tensor(1.3), torch.tensor(0.9))
# dloss = loss_function(ans, torch.tensor(1), torch.tensor(1 / 1.3), torch.tensor(0.9))
# print(ans, uloss, dloss)

num_epochs = 2000
answer = analytical_answer(up_prices, down_prices, exercise_price)
print(answer)
min_loss = loss_function(answer, previous_prices, next_prices, exercise_price)
print(min_loss)

for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = policy(up_prices, down_prices, exercise_price)
    
    # Calculate the loss
    loss = loss_function(outputs, previous_prices, next_prices, exercise_price)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch + 1} loss:', float(loss))
        # print(torch.linalg.vector_norm(outputs - answer))


outputs = policy(up_prices, down_prices, exercise_price)
print(outputs.tolist())