from math import exp
from random import random

'''
This file create an optimal agent to the problem under action discretization that one can only invest all the wealth or none at all
A tree structure is used to store all the possible transition values hence find the optimal action
'''

# Storing the action decision nodes in the tree for backward induction
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

# Storing the result decision nodes in the tree for backward induction
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

# Helper function for computing utility
def compute_utility(w):
    return 1 - exp(-alpha * w) / alpha


# Class for node that brach out two possible action
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
        self.rd2 = resultDecisionNode(self.W, self.t, 0)
        result_decisions[self.t].append(self.rd2)
        self.rd2.grow_tree()

    def max_q(self):
        if self.t == 11:
            return compute_utility(self.W)
        if self.q1 > self.q2:
            return self.q1
        else:
            return self.q2

# Class for node that brach out two possible outcome after an action is taken
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
        w1 = self.x * self.W * (1 + a) + self.W * (1 - self.x) * (1 + r)
        w2 = self.x * self.W * (1 + b) + self.W * (1 - self.x) * (1 + r)
        # w1 = self.x * (1 + a) + (self.W - self.x) * (1 + r)
        # w2 = self.x * (1 + b) + (self.W - self.x) * (1 + r)
        self.ad1 = actionDecisionNode(w1, self.t + 1)
        action_decisions[self.t + 1].append(self.ad1)
        self.ad1.grow_tree()
        self.ad2 = actionDecisionNode(w2, self.t + 1)
        action_decisions[self.t + 1].append(self.ad2)
        self.ad2.grow_tree()

    def expected_q(self):
        global p
        return p * self.e1 + (1 - p) * self.e2



# Agent with discrete action that either buy risky asset with all wealth or do nothing
# Optimal solution is found using backward induction on a tree structure
class TreeAgent:

    def __init__(self) -> None:

        # Create the tree strcture
        root = actionDecisionNode(W0, 1)
        action_decisions[1] = [root]
        root.grow_tree() 

        # Backward induction on the tree structure
        for i in range(10):
            j = 10 - i
            for node in result_decisions[j]:
                node.e1 = node.ad1.max_q()
                node.e2 = node.ad2.max_q()
            for node in action_decisions[j]:
                node.q1 = node.rd1.expected_q()
                node.q2 = node.rd2.expected_q()

        self.root = root
        self.current = root
        self.now = 1

    # Perform the best action according the transition values stored in the trees
    def perform_action(self):
        if self.current.q1 > self.current.q2:
            self.current = self.current.rd1
            return 1
        else:
            self.current = self.current.rd2
            return 0
    
    # Perform a random action
    def perform_random_action(self):
        if random() > 0.5:
            self.current = self.current.rd1
            return 1
        else:
            self.current = self.current.rd2
            return 0
    
    # Traverse the tree to the next state with the luck variable to indicate if the stock goes up or down
    def transition(self, luck):
        if luck == 1:
            self.current = self.current.ad1
        else:
            self.current = self.current.ad2

    # Reset the current node to root for another traverse
    def reset(self):
        self.current = self.root


# Environment properties
W0 = 1
T = 10
a = 0.3
b = -0.1
p = 0.5
r = 0.05
alpha = 1

# Create a tree and test its performance
if __name__ == '__main__':
    ta = TreeAgent()
    lucks = [int(p > random()) for _ in range(10)]
    lucks = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    print('Transition path (1 = up, 0 = down) -', lucks)
    for l in lucks:
        a = ta.perform_random_action()
        ta.transition(l)
    print('Reward from random agent:', compute_utility(ta.current.W))
    
    ta.reset()
    actions = []
    for l in lucks:
        a = ta.perform_action()
        actions.append(a)
        ta.transition(l)
    print('Reward from optimal agent:', compute_utility(ta.current.W))
    print('Actions taken by optimal agent:', actions)