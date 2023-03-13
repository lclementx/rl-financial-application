from wealth_state import WealthState
from investment_action import InvestmentAction
from collections import defaultdict
from typing import Dict, Sequence
import numpy as np

'''
Given an intiial state and all available actions, risky return distribution and risk free rate,
create all the possible state + action pairs and the associated probability. I attach the next
states as well given the action to make computation afterwards easier.

Return:
state: {
    action1:{next_state1:probability,next_state2:probability}
    action2:{next_state3:probability,next_state4:probability}
}
'''
def get_all_state_actions(
    iterations, 
    available_actions, 
    initial_state, 
    risky_return_distribution,
    risk_free_rate
) \
-> Sequence[Dict[WealthState, Dict[WealthState, float]]]:
    state_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    current_states = [initial_state]
    for i in range(iterations):
        next_states_list = set()
        for state in current_states:
            for action , action_prob in available_actions.items():
                for risky_return, risky_return_prob in risky_return_distribution.items():
                    next_state = action.action_to_next_state(state,risky_return,risk_free_rate)
                    state_tree[state][action][next_state]+= risky_return_prob * action_prob
                    next_states_list.add(next_state)

        current_states = next_states_list
    return state_tree

'''
Given all the state_action probabilities above, try to calculate each state-action's reward given a specific reward function.

Return:
state: {
    action1:reward1
    action2:reward2
}
'''
def get_state_action_value_map(state_actions, reward_func, gamma,alpha,iterations):
    
    def get_reward(state,action):
        next_states_probablity = state_actions[state][action]
        total_reward = 0
        for next_state, action_return_probability in next_states_probablity.items():
            if next_state.isTerminal():
                utility = reward_func(next_state.wealth,alpha)
                total_reward += utility * action_return_probability
                # print(f'Utility: {utility}, Action: {action}, Prob: {action_return_probability}, Total reward: {total_reward}')
            else:
                next_state_actions = state_actions[next_state]
                for next_action, _ in next_state_actions.items():
                    total_reward += get_reward(next_state,next_action) * action_return_probability
        
        # print(f'State: {state} , Action: {action}, Reward: {total_reward}')
        return total_reward
            
    state_action_value_map = defaultdict(lambda: defaultdict(float))
    for i in reversed(range(iterations)):
        states_at_t = [state for state in state_actions.keys() if state.time == i]
        for (state_at_t) in states_at_t:
            next_state_actions = state_actions[state_at_t]
            discount = gamma ** (iterations - i)
            for action, next_states in next_state_actions.items():
                reward = get_reward(state_at_t,action)
                state_action_value_map[state_at_t][action] = discount * reward 
    
    return state_action_value_map

'''
Once I have all the rewards for each state-action, my optimal policy for each state is to pick the action that will provide the
highest reward.
'''
def retrieve_optimal_policy_from_values(state_action_value_map):
    policy = defaultdict(lambda: defaultdict(float))
    for state, action_values in state_action_value_map.items():
        action = max(action_values, key=action_values.get)
        value = action_values[action]
        policy[state][action]=1
    return policy

'''
Run a given policy for a given initial state to look at what the results will look like.
'''
def execute_policy(policy,initial_state,risky_return_dist,risk_free_rate):
    state = initial_state
    termination = state.termination_time
    for _ in range(termination):
        action_probs = policy[state]
        action = max(action_probs, key=action_probs.get)
        risky_return = np.random.choice([*risky_return_dist.keys()], 
                                     p=[*risky_return_dist.values()])
        next_state = action.action_to_next_state(state,risky_return,risk_free_rate)
        print(f'State: {state}, Action: {action}')
        state = next_state