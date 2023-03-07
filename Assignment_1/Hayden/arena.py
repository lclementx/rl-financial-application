from random import random
from math import exp
from agent import random_agent
from agent import QNetwork
from agent import PNetwork

import torch

# Get the next state and rewards
def get_next_state(t, Wt, xt, a, b, p, r, alpha=1):
    a_W = xt * (a - r) + Wt * (1 + r)
    b_W = xt * (b - r) + Wt * (1 + r)
    expected_reward = p * -exp(-alpha * a_W) / alpha + (1 - p) * -exp(-alpha * b_W) / alpha
    if random() < p:
        Yt = a
    else :
        Yt = b
    next_W = xt * (Yt - r) + Wt * (1 + r)
    if t == 1:
        reward = -exp(-alpha * next_W) / alpha
    else:
        reward = 0
    return next_W, reward, expected_reward


# Solve last step for method verification
def solve_last_step(Wt, a, b, p, r, alpha=1):
    step = 0.1
    point = 0.5
    for _ in range(1000):
        xt = Wt * point
        grad = p * (a - r) * exp(-alpha * xt * (a - r)) + (1 - p) * (b - r) * exp(-alpha * xt * (b - r))
        point += grad * step
    return Wt * point


def generate_sample(T, W, a, b, p, r, N=1000, alpha=1):
    
    states = []
    rewards = []
    for _ in range(N):
        Wt = W
        state = []
        for t in range(T):
            xt = random_agent(Wt)
            next_W, reward, _ = get_next_state(t + 1, Wt, xt, a, b, p, r, alpha)
            state.append([t + 1, Wt, xt])
            if t + 1 == T:
                rewards.append(reward)
            Wt = next_W
        states.append(state)
    return states, rewards


def train_Q_network(states, rewards, T=10, N=1000):
    Q = QNetwork()
    X = torch.tensor(states).view(-1, 3)
    y = torch.tensor(rewards).repeat_interleave(T)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.SGD(Q.parameters(), lr=0.00001)
    for i in range(N):
        pred = Q(X)
        loss = criterion(pred.view(len(X)), y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (i + 1) % 50 == 0:
            print(f'Epoch {i + 1} loss: {float(loss)}')
    return Q


def train_P_network(Q, states, N=1000):
    P = PNetwork()
    S = torch.tensor(states).view(-1, 3)[:, :2]
    optim = torch.optim.SGD(P.parameters(), lr=0.002)
    for i in range(N):
        a = P(S)
        X = torch.cat([S, a], dim=1)
        loss = - Q(X).sum() / len(S)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (i + 1) % 50 == 0:
            print(f'Epoch {i + 1} loss: {float(loss)}')
    return P



W = 1
a = 0.2
b = -0.1
p = 0.6
r = 0.05
alpha = 3

states, rewards = generate_sample(1, W, a, b, p, r, 3000, alpha)
Q = train_Q_network(states, rewards, 1, 100)
print('---------')
# P = train_P_network(Q, states, 1000)
print(Q.estimate([1], [1], [0.6]))
print(get_next_state(1, 1, 0.6, a, b, p, r, alpha))
print(Q.estimate([1], [1], [1.258]))
print(get_next_state(1, 1, 1.258, a, b, p, r, alpha))
print(Q.estimate([1], [1], [3]))
print(get_next_state(1, 1, 3, a, b, p, r, alpha))
# print(P.estimate([1], [1]))

Wt = W
print(solve_last_step(Wt, a, b, p, r, alpha))
print(sum(rewards)/ len(rewards))

# rr = 0
# sr = 0
# for _ in range(100):
#     rx = random_agent(Wt)
#     nW, reward = get_next_state(1, Wt, rx, a, b, p, r, alpha)
#     rr += (reward + 1)
#     sx = solve_last_step(Wt, a, b, p, r, alpha)
#     nW, reward = get_next_state(1, Wt, sx, a, b, p, r, alpha)
#     sr += (reward + 1)
# print(rr, sr)