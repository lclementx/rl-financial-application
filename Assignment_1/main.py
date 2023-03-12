from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import binom
import numpy as np
from typing import Dict, Sequence
import itertools
import random
import operator
from operator import itemgetter
from wealth_state import WealthState
from investment_action import InvestmentAction
from transitions import *
from utility import *
from mathematical_solution import *

ITERATIONS=10
GAMMA=1
COEFFICIENT_OF_CARA=1
INVESTMENT_LIMIT_MAX=1
INVESTMENT_LIMIT_MIN=0
SPLITS=2
RISK_FREE_RATE=0.05
PROBABILITY_PRICE_UP=0.6 #Probability of risky asset increasing
PRICE_A=0.3
PRICE_B=-0.1
INITIAL_WEALTH=1
RISKY_RETURN_DISTRIBUTION={PRICE_A:PROBABILITY_PRICE_UP, PRICE_B:1-PROBABILITY_PRICE_UP}

if __name__ == '__main__':
    initial_state = WealthState(time=0,wealth=INITIAL_WEALTH,termination_time=ITERATIONS)
    all_actions = InvestmentAction.get_all_actions(INVESTMENT_LIMIT_MAX,INVESTMENT_LIMIT_MIN,SPLITS)
    initial_action_probs = {a: 1/len(all_actions) for a in all_actions}
    all_state_actions = get_all_state_actions(ITERATIONS,initial_action_probs,initial_state,RISKY_RETURN_DISTRIBUTION,RISK_FREE_RATE)
    state_action_value_map=get_state_action_value_map(all_state_actions,cara_func,GAMMA,COEFFICIENT_OF_CARA,ITERATIONS)
    policy = retrieve_optimal_policy_from_values(state_action_value_map)
    print('Attempt 1: For Discrete Time, Discrete Action (+1,0) - calculate value for each state-action to find the optimal step')
    print('----------------------------------------------------------------------------------------------------------------')
    execute_policy(policy,initial_state,RISKY_RETURN_DISTRIBUTION,RISK_FREE_RATE)
    print('Attempt 2: Use a tree structure to support backward induction with code executed from tree.py')
    import tree
    print('----------------------------------------------------------------------------------------------------------------')
    print('Attempt 3: Use Monte Carlo method and regression to estimate Q function for continous state and action with code executed from train.py')
    print('**It may take a few minutes to complete**')
    # import train
    print('----------------------------------------------------------------------------------------------------------------')
    print('Attempt 4: Use Mathematical solution - derivation of the solution attached as part of the respository')
    print('----------------------------------------------------------------------------------------------------------------')
    plot_analytical_solution(ITERATIONS,INITIAL_WEALTH,PROBABILITY_PRICE_UP,PRICE_A,PRICE_B,RISK_FREE_RATE,COEFFICIENT_OF_CARA)

