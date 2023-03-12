from wealth_state import WealthState
from investment_action import InvestmentAction
from collections import defaultdict
from typing import Dict, Sequence

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

def retrieve_optimal_policy_from_values(state_action_value_map):
    policy = defaultdict(lambda: defaultdict(float))
    for state, action_values in state_action_value_map.items():
        action = max(action_values, key=action_values.get)
        value = action_values[action]
        policy[state][action]=1
    return policy