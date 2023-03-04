from random import random
from math import exp
from agent import random_agent


# Get the next state and rewards
def get_next_state(t, Wt, xt, a, b, p, r, alpha=1):
    if random() < p:
        Yt = a
    else :
        Yt = b
    next_W = xt * (Yt - r) + Wt * (1 + r)
    if xt > Wt or Wt < 0:
        reward = -1
    elif t == 1:
        reward = -exp(-alpha * next_W) / alpha
    else:
        reward = 0
    return next_W, reward


# Solve last step for method verification
def solve_last_step(Wt, a, b, p, r, alpha=1):
    step = 0.01
    point = 0.5
    for _ in range(5000):
        xt = Wt * point
        grad = p * (a - r) * exp(-alpha * xt * (a - r)) + (1 - p) * (b - r) * exp(-alpha * xt * (b - r))
        if grad > 0:
            point += step
        else:
            point -= step
    return Wt * point


Wt = 1
a = 0.2
b = -0.1
p = 0.2
r = 0.05
alpha = 1

print(solve_last_step(Wt, a, b, p, r, alpha))

rr = 0
sr = 0
for _ in range(100):
    rx = random_agent(Wt)
    nW, reward = get_next_state(1, Wt, rx, a, b, p, r, alpha)
    rr += (reward + 1)
    sx = solve_last_step(Wt, a, b, p, r, alpha)
    nW, reward = get_next_state(1, Wt, sx, a, b, p, r, alpha)
    sr += (reward + 1)
print(rr, sr)