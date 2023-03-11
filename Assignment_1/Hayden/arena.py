from random import random
from math import exp
from regressor import random_agent, QRegressor

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
    if t == 10:
        reward = -exp(-alpha * next_W) / alpha
    else:
        reward = 0
    return next_W, reward


# Solve last step for method verification
def solve_last_step(Wt, a, b, p, r, alpha=1):
    step = 0.1
    point = 0.5
    for _ in range(1000):
        xt = Wt * point
        grad = p * (a - r) * exp(-alpha * xt * (a - r)) + (1 - p) * (b - r) * exp(-alpha * xt * (b - r))
        point += grad * step
    return Wt * point


def generate_sample(Q, W, a, b, p, r, T, N=1000, alpha=1, epsilon=1):
    
    states = []
    rewards = []
    for _ in range(N):
        Wt = W
        state = []
        for t in range(T):
            if random() < epsilon:
                xt = random_agent(Wt)
            else:
                xt = Q[t].find_max(Wt)
            next_W, reward = get_next_state(t + 1, Wt, xt, a, b, p, r, alpha)
            state.append([Wt, xt])
            Wt = next_W
            if reward != 0:
                rewards.append(reward)
        states.append(state)
    return states, rewards


def train_regressor(states, rewards, T=10, N=1000):
    Q = {}
    W = {}
    X = {}
    R = {}
    for t in range(T):
        Q[t] = QRegressor()
        W[t] = []
        X[t] = []
    for i in range(N):
        for t in range(T):
            Wt, xt = states[i][t]
            W[t].append(Wt)
            X[t].append(xt)
            R[t].append(rewards[i])
    for t in range(T):
        Q[t].train(W, X)
    return Q
        

W = 1
a = 0.3
b = -0.1
p = 0.5
r = 0.05
alpha = 1

states, rewards = generate_sample(1, W, a, b, p, r, 3000, alpha)
model = train_regressor(states, rewards, 1, 100)
