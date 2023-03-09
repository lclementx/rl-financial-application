from math import exp
from random import random

action_decisions = {
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
    11: []
}

result_decisions = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: []  
}

W0 = 0
p = 0.5
a = 0.3
b = -0.1
r = 0.05
alpha = 1

def compute_utility(w):
    return 1 - exp(-alpha * w) / alpha


class actionDecisionNode:

    def __init__(self, W, t):
        self.W = W
        self.t = t
        self.rd1 = None
        self.q1 = None
        self.rd2 = None
        self.q2 = None

    def grow_tree(self):
        if self.t == 11:
            return
        self.rd1 = resultDecisionNode(self.W, self.t, 1)
        result_decisions[self.t].append(self.rd1)
        self.rd1.grow_tree()
        self.rd2 = resultDecisionNode(self.W, self.t, -1)
        result_decisions[self.t].append(self.rd2)
        self.rd2.grow_tree()

    def max_q(self):
        if self.t == 11:
            return compute_utility(self.W)
        if self.q1 > self.q2:
            return self.q1
        else:
            return self.q2


class resultDecisionNode:

    def __init__(self, W, t, x):
        self.W = W
        self.t = t
        self.x = x
        self.ad1 = None
        self.e1 = None
        self.ad2 = None
        self.e2 = None

    def grow_tree(self):
        w1 = self.x * (1 + a) + (self.W - self.x) * (1 + r)
        w2 = self.x * (1 + b) + (self.W - self.x) * (1 + r)
        self.ad1 = actionDecisionNode(w1, self.t + 1)
        action_decisions[self.t + 1].append(self.ad1)
        self.ad1.grow_tree()
        self.ad2 = actionDecisionNode(w2, self.t + 1)
        action_decisions[self.t + 1].append(self.ad2)
        self.ad2.grow_tree()

    def expected_q(self):
        global p
        return p * self.e1 + (1 - p) * self.e2


class TreeAgent:

    def __init__(self) -> None:
        root = actionDecisionNode(W0, 1)
        action_decisions[1] = [root]
        root.grow_tree()

        for i in range(10):
            j = 10 - i
            for node in result_decisions[j]:
                node.e1 = node.ad1.max_q()
                node.e2 = node.ad2.max_q()
            for node in action_decisions[j]:
                node.q1 = node.rd1.expected_q()
                node.q2 = node.rd2.expected_q()

        self.current = root
        self.now = 1

    def perform_action(self):
        if self.current.q1 > self.current.q2:
            self.current = self.current.rd1
            return 1
        else:
            self.current = self.current.rd2
            return 0
        
    def perform_random_action(self):
        if random() > 0.5:
            self.current = self.current.rd1
            return 1
        else:
            self.current = self.current.rd2
            return 0
    
    def transition(self, luck):
        if luck == 1:
            self.current = self.current.ad1
        else:
            self.current = self.current.ad2


ta = TreeAgent()
lucks = [int(random() > p) for _ in range(10)]
# lucks = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
actions = []
for l in lucks:
    a = ta.perform_action()
    actions.append(a)
    ta.transition(l)
print(lucks)
print(actions)
print(ta.current.W)
print(compute_utility(ta.current.W))