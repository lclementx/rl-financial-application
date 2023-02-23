import operator
import numpy as np
from dataclasses import dataclass
from typing import Generic, Callable, TypeVar, Iterable,\
Optional, Mapping, Tuple, Sequence
from distribution import Choose
from state import *
from markov_decision_process import *
from iterate import *
from itertools import groupby

A = TypeVar('A')
S = TypeVar('S')

DEFAULT_TOLERANCE = 1e-5

V = Mapping[NonTerminal[S], float]

#Evaluate the Markov Reward Process V = R + gamma * P_r * V
def evaluate_mrp(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
)-> Iterator[np.ndarray]:
    def update(v: np.ndarray) -> np.array:
        return mrp.reward_function_vec + gamma * mrp.get_transition_matrix().dot(v)
    
    v_0: np.ndarray = np.zeros(len(mrp.non_terminal_states))
    return iterate(update, v_0)

#Used to compare the arrays given a tolerance
def almost_equal_np_arrays(
    v1: np.ndarray,
    v2: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    return max(abs(v1 - v2)) < tolerance

#Helps to visualize the mrp results
def evaluate_mrp_result(
    mrp:FiniteMarkovRewardProcess[S],
    gamma: float
) -> V[S]:
    v_star: np.ndarray = converged(
        evaluate_mrp(mrp, gamma=gamma),
        done=almost_equal_np_arrays
    )
    return {s: v_star[i] for i, s in enumerate(mrp.non_terminal_states)}

#Extended Value Function - for non-terminal states return the value function, Terminal = 0.
def extended_vf(v: V[S], s: State[S]) -> float:
    def non_terminal_vf(st: NonTerminal[S], v=v) -> float:
        return v[st]
    return s.on_non_terminal(non_terminal_vf,0.0) #second arg is the default value

#Given a FintieMarkovDecisionProcess and Value function, find the greedy policy
#Greedy Policy -  taking the action that will yield the highest value :)
def greedy_policy_from_vf(
    mdp:FiniteMarkovDecisionProcess[S, A],
    vf: V[S],
    gamma: float
) -> FiniteDeterministicPolicy[S, A]:
    greedy_policy_dict: Dict[S, A] = {}
    
    for s in mdp.non_terminal_states:
        #q_values = for a given state and action, get apply reward + gamma * value function()
        q_values: Iterator[Tuple[A, float]] = \
            ((a, mdp.mapping[s][a].expectation(
                lambda s_r: s_r[1] + gamma * extended_vf(vf, s_r[0])
            )) for a in mdp.actions(s))
        
        #Find the action that gives the max value. itemgetter gets the reward from the Tuple[Action, float].
        greedy_policy_dict[s.state] =\
            max(q_values, key=operator.itemgetter(1))[0]
    return FiniteDeterministicPolicy(greedy_policy_dict)
    
def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    matrix_method_for_mrp_eval: bool = False
) -> Iterator[Tuple[V[S], FinitePolicy[S, A]]]:
    
    #Define an update function
    #Start with a value function and policy: 
    #1. evaluate the policy to get a value function
    #2. with the value function, use the greedy policy from value function to find the improved policy
    def update(vf_policy: Tuple[V[S], FinitePolicy[S, A]])\
        -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        policy_vf: V[S] = {mrp.non_terminal_states[i]: v for i , v in
                           enumerate(mrp.get_value_function_vec(gamma))}\
            if matrix_method_for_mrp_eval else evaluate_mrp_result(mrp, gamma)
        #Use the greedy policy from value function to find the improved policy
        improved_pi: FiniteDeterministicPolicy[S, A] = greedy_policy_from_vf(
            mdp,
            policy_vf,
            gamma
        )
        return policy_vf, improved_pi
    
    #Instantiate the first value function with all non terminal states and value = 0
    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    #Instantiate the first policy by mapping state --> all actions as a probability distribution with equal chance
    pi_0: FinitePolicy[S, A] = FinitePolicy(
        {s.state: Choose(mdp.actions(s)) for s in mdp.non_terminal_states}
    )
    return iterate(update, (v_0, pi_0))

#Again - define a function for value function comparison
def almost_equal_vf_pis(
    x1: Tuple[V[S], FinitePolicy[S, A]],
    x2: Tuple[V[S], FinitePolicy[S, A]]
) -> bool:
    return max(
        abs(x1[0][s] - x2[0][s]) for s in x1[0]
    ) < DEFAULT_TOLERANCE

def policy_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    return converged(policy_iteration(mdp, gamma), done=almost_equal_vf_pis)

#Value iteration - for a given state, find the action that give you the highest value
#Keep doing this until your value function converges i.e. the v_t - v_t-1 < threshold
def value_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Iterator[V[S]]:
    def update(v: V[S]) -> V[S]:
        #
        return {s: max(mdp.mapping[s][a].expectation(
            lambda s_r: s_r[1] + gamma * extended_vf(v, s_r[0])
        ) for a in mdp.actions(s)) for s in v}
    
    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    return iterate(update, v_0)

def almost_equal_vfs(
    v1: V[S],
    v2: V[S],
    tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    return max(abs(v1[s] - v2[s]) for s in v1) < tolerance

def value_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    opt_vf: V[S] = converged(
        value_iteration(mdp, gamma),
        done=almost_equal_vfs
    )
    opt_policy: FiniteDeterministcPolicy[S, A] = greedy_policy_from_vf(
        mdp,
        opt_vf,
        gamma
    )
    return opt_vf, opt_policy