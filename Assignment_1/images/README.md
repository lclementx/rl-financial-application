The mathematical proof focuses on CARA utility. The goal is to maximise: $\mathbb{E}[\gamma^{T-t} * \frac{1 - e^{-\alpha W_t}}{\alpha}|(t, W_t)]$

In this expectation, $\gamma$ and $\frac{1}{\alpha}$ are actually constants so the focus of the derivation is to maximise $\mathbb{E}[\frac{-e^{-\alpha W_t}}{\alpha}|(t, W_t)]$.

Wealth is definedto be: $W_t+1 = x_t(Y_t - r) + W_t(1 + r)$, where $x_t$ is the risky asset allocation, $Y_t$ is the return on risky_asset at time t and r is the risk free return.

$Y_t$ in our case follows a Binomial Distribution: $A$ return with probability $P$, $B$ return with probability $1-P$

Approach:
1. Using the Moment Generating Function for Binomial Distribution, retrieve an expression for the minimum (which is the maximum when negated)
2. Assume the solution will be in the from $-b_{t+1}e^{c_{t+1}W_{t}}$ then find a recursive form for $c_{t+1}$ and $b_{t+1}$
3. Also from 2, get an expression for $x_t$ in terms of $b_{t+1}$ and $c_{t+1}$ - this is the expression that will tell us what the optimal allocation should be
4. Use the fact that at time T, CARA utility = $\mathbb{E}[\frac{-e^{-\alpha W_T}}{\alpha})]$ to derive what $b_t$ and $c_t$ should be.
