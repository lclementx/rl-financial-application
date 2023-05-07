import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

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
RISK_FREE_RATE = 0.03
RISK_FREE_FACTOR = 1 + RISK_FREE_RATE
INITIAL_PRICE = 1
EXERCISE_PRICE = 1

# Store all possible stock price transition path in a matrix
price_paths = []
for transition_path in itertools.product([UP_FACTOR, DOWN_FACTOR], repeat=10):
    current_price = INITIAL_PRICE
    path = []
    for transition in transition_path:
        path.append(current_price)
        current_price *= transition
    path.append(current_price)
    price_paths.append(path)
price_paths = torch.tensor(price_paths, dtype=torch.float32).transpose(0, 1)


def variance_loss(policies, n=10):

    up_factor = torch.full((2 ** n,), UP_FACTOR, dtype=torch.float32)
    down_factor = torch.full((2 ** n,), DOWN_FACTOR, dtype=torch.float32)
    risk_free_factor = torch.full((2 ** n,), RISK_FREE_FACTOR, dtype=torch.float32)
    exercise_price = torch.full((2 ** n,), EXERCISE_PRICE, dtype=torch.float32)

    cash = torch.full((2 ** n,), 0, dtype=torch.float32)
    holding = torch.full((2 ** n,), 0, dtype=torch.float32)

    for i, policy in enumerate(policies):

        current_price = price_paths[i].detach()

        # Record situation the beginning of the timestep
        up_price = (current_price * up_factor).detach()
        down_price = (current_price * down_factor).detach()

        # Perform re-balancing
        hedge_ratio = policy(up_price, down_price, exercise_price, cash, risk_free_factor).view(-1)
        cash = cash - (hedge_ratio - holding) * current_price
        holding = hedge_ratio

        cash = cash * risk_free_factor.detach()

    current_price = price_paths[10].detach()
    final_portfolio_values = cash + holding * current_price - torch.where(exercise_price > current_price, exercise_price - current_price, torch.zeros_like(current_price))
    return final_portfolio_values.var(), final_portfolio_values.mean()

policies = [Policy() for _ in range(10)]
all_params = []
for i, policy in enumerate(policies):
    all_params.extend(list(policy.parameters()))
                      
optimizer = optim.SGD(all_params, lr=0.1)

chart_x = []
chart_y = []

num_epochs = 15000
for epoch in range(num_epochs):

    # Zero the gradients
    optimizer.zero_grad()
    
    # Calculate the loss
    variance, mean = variance_loss(policies)

    # Backward pass
    variance.backward()

    # Update the weights
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1} variance:', format(float(variance), '.8f'))
        chart_x.append(epoch + 1)
        chart_y.append(float(variance))

print(f'Hedging inferred option price:', format(-mean / RISK_FREE_FACTOR ** 10, '.8f'))

RISK_NEUTRAL_PROB = (RISK_FREE_FACTOR - DOWN_FACTOR) / (UP_FACTOR - DOWN_FACTOR)
paths_prob = []
for transition_prob in itertools.product([RISK_NEUTRAL_PROB, 1 - RISK_NEUTRAL_PROB], repeat=10):
    path_prob = 1
    for p in transition_prob:
        path_prob *= p
    paths_prob.append(path_prob)

# Save the model
for i, policy in enumerate(policies):
    torch.save(policy.state_dict(), f'parameters/european_t{i}.pt')

# Calculate the expected value option price for comparison
paths_prob = torch.tensor(paths_prob, dtype=torch.float32)
exercise_price = torch.full((2 ** 10,), EXERCISE_PRICE, dtype=torch.float32)
paths_payout = torch.where(exercise_price > price_paths[10], exercise_price - price_paths[10], torch.zeros_like(exercise_price))
option_option = (paths_prob * paths_payout).sum() / RISK_FREE_FACTOR ** 10
print(f'Expected value option price:', format(option_option, '.8f'))

# Create a line chart to show convergence as training progresses
plt.plot(chart_x, chart_y)
plt.title('Variance on Portfolio Values (European)')
plt.xlabel('Steps')
plt.show()
