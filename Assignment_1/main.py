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

ITERATIONS=4
GAMMA=1
COEFFICIENT_OF_CARA=1
INVESTMENT_LIMIT_MAX=1
INVESTMENT_LIMIT_MIN=-1
SPLITS=3
RISK_FREE_RATE=0.01
PROBABILITY_PRICE_UP=0.7 #Probability of risky asset increasing
PRICE_A=0.2
PRICE_B=-0.1
INITIAL_WEALTH=0
RISKY_RETURN_DISTRIBUTION={PRICE_A:PROBABILITY_PRICE_UP, PRICE_B:1-PROBABILITY_PRICE_UP}

def execute_policy(policy,initial_state,risky_return_dist,risk_free_rate):
    state = initial_state
    termination = state.termination_time
    for _ in range(termination):
        action_probs = policy[state]
        action = max(action_probs, key=action_probs.get)
        risky_return = np.random.choice([*RISKY_RETURN_DISTRIBUTION.keys()], 
                                     p=[*RISKY_RETURN_DISTRIBUTION.values()])
        next_state = action.action_to_next_state(state,risky_return,risk_free_rate)
        print(f'State: {state}, Action: {action}')
        state = next_state

if __name__ == '__main__':
    initial_state = WealthState(time=0,wealth=INITIAL_WEALTH,termination_time=ITERATIONS)
    all_actions = InvestmentAction.get_all_actions(INVESTMENT_LIMIT_MAX,INVESTMENT_LIMIT_MIN,SPLITS)
    initial_action_probs = {a: 1/len(all_actions) for a in all_actions}
    all_state_actions = get_all_state_actions(ITERATIONS, initial_action_probs, initial_state,RISKY_RETURN_DISTRIBUTION,RISK_FREE_RATE)
    state_action_value_map=get_state_action_value_map(all_state_actions,cara_func,GAMMA,ITERATIONS)
    policy = retrieve_optimal_policy_from_values(state_action_value_map)
    execute_policy(policy,initial_state,RISKY_RETURN_DISTRIBUTION,RISK_FREE_RATE)
