### Assignment: Consider the discrete-time asset allocation example in section 8.4 of Rao and Jelvis. Suppose the single-time-step return of the risky asset from time t to t+1 as Yt = a, prob = p and b, prob = (1-p). Suppose that T=10, use the TD method to find the Q function and hence the optimal strategy

Several approaches were taken to tackle the problem above:
1. By considering discrete time and discrete action space (+1/+0) - enumerate all possible states, calculate the total reward for each state-action and thus retrieve the optimal policy by following the state actions witht the highest reward
2. Use a tree structure to support backward induction with code executed from tree.py
3. Use Monte Carlo method and regression to estimate Q function for continous state and action with code executed from train.py
4. Use Mathematical solution - derivation of the solution attached as part of the respository
