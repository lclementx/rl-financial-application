from __future__ import annotations
from typing import Iterator, Tuple, TypeVar, Sequence, List
from operator import itemgetter
import numpy as np

from distribution import Distribution
from function_approx import FunctionApprox
from iterate import iterate
from markov_decision_process import (FiniteMarkovRewardProcess, MarkovRewardProcess, RewardTransition)
from state import State, NonTerminal
from markov_decision_process import (FiniteMarkovDecisionProcess, MarkovDecisionProcess, StateActionMapping)
from policy import DeterministicPolicy

S = TypeVar('S')
A = TypeVar('A')

'''
#In order to do value function approximation, we can't
#enumerate all states, so we define a probability
#distribution to sample states from (Non Terminal States)
'''
NTStateDistribution = Distribution[NonTerminal[S]]

'''
#Sample pairs (<- plural) of next state = s', reward = r
#from a given non-terminal state s, and calcualte the
#expectation E[r + gamma * V(s') by averaging r + gamma * V(s') across the sampled pairs i.e. [(r1 + gamma * V(s'_1) ) + r2 + gamma * V(s'_2) ] /2 (2 samples here)
'''
ValueFunctionApprox = FunctionApprox[NonTerminal[S]]


'''
Triplet used for backward induction:
|MarkovRewardProcess| - each time step has its own MRP representation of transitions from non-terminal states s in a time step t to the (state s' and reward r) paris in the next time step t+1
|ValueFunctionApprox| - to capture the ValueFunctionApproximation for the time step
|NTStateDistribution| - A sampling probability distribution of non terminal states in the time step
'''
MRP_FuncApprox_Distribution = Tuple[MarkovRewardProcess[S],
                                    ValueFunctionApprox[S],
                                    NTStateDistribution[S]]

'''
Triplet for Approximate Value Iteration
'''
MDP_FuncApprox_Distribution = Tuple[MarkovDecisionProcess[S, A],
                                    ValueFunctionApprox[S],
                                    NTStateDistribution[S]]
'''
For Q-Value Function Approximation
'''
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]

MDP_FuncApproxQ_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    QValueFunctionApprox[S, A],
    NTStateDistribution[S]]

'''
The sampled list of non-terminal states = x and the associated sampled expectations = y (since it is E[y|x]) These (x,y) pairs are sued to update the approximation of the Value Function in each iteration (producing a new instance of ValueFunctionApprox using its update method

#Similar to the one defined in dynamic programming - where you don't know whether you will end up at a Terminal state. Therefore extended vf will return 0. if you hit terminal, else KEEP GOINGGGG :)
'''
def extended_vf(vf: ValueFunctionApprox[S], s: State[S]) -> float:
    return s.on_non_terminal(vf, 0.0)

def evaluate_finite_mrp(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    approx_0: ValueFunctionApprox[S]
)-> Iterator[ValueFunctionApprox[S]]:
    
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        vs: np.ndarray = v.evaluate(mrp.non_terminal_states)
        updated: np.ndarray = mrp.reward_function_vec + gamma * mrp.get_transition_matrix().dot(vs)
        
        return v.update(zip(mrp.non_terminal_states, updated))
    
    return iterate(update, approx_0)
                            

#Find the approximate value function - Policy Evaluation Problem
def evaluate_mrp(
    mrp: MarkovRewardProcess[S], #MRP implied by policy - pi
    gamma: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int
) -> Iterator[ValueFunctionApprox[S]]:
    
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        #Sample states from NonTerminal states :)
        nt_states: Sequence[NonTerminals[S]] = \
            non_terminal_states_distribution.sample_n(num_state_samples)
        
        #the return is, given a s' and r, r * gamma * V(s')
        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + gamma * extended_vf(v, s1)
        
        return v.update(
            [(s, mrp.transition_reward(s).expectation(return_))
             for s in nt_states]
        )
    
    #The code that calls the iterator and decide when to stop this way :)
    return iterate(update, approx_0)
               
def value_iteration(
    mdp:MarkovDecisionProcess[S, A],
    gamma: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int
) -> Iterator[ValueFunctionApprox[S]]:
    
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        nt_states: Sequence[NonTerminal[S]] = \
            non_terminal_states_distribution.sample_n(num_state_samples)
        
        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + gamma * extended_vf(v, s1)
        
        #For each non-terminal state, find the action that will return the maximum
        #expected return
        return v.update(
            [(s, max(mdp.step(s,a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in nt_states]
        )
    
    return iterate(update, approx_0)

'''
This is the Approximate Policy Evaluation for Finite-Horizon (Prediction Problem)
'''
def backward_evaluate(
    mrp_f0_mu_triples: Sequence[MRP_FuncApprox_Distribution[S]],
    gamma: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[ValueFunctionApprox[S]]:
    v: List[ValueFunctionApprox[S]] = []
    
    for i, (mrp, approx0, mu) in enumerate(reversed(mrp_f0_mu_triples)):
        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            #Note we are going backwards (i-1) here
            return r + gamma * (extended_vf(v[i-1],s1) if i > 0 else 0.)
        
        v.append(
            approx0.solve(
                [(s, mrp.transition_reward(s).expectation(return_))
                 for s in mu.sample_n(num_state_samples)],
                error_tolerance
            )
        )
        
    return reversed(v)

def back_opt_vf_and_policy(
    mdp_f0_mu_triples: Sequence[MDP_FuncApprox_Distribution[S, A]],
    gamma: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    vp: List[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]] = []
    
    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):
        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + gamma * (extended_vf(vp[i-1][0], s1) if i > 0 else 0)
        
        this_v = approx0.solve(
            [(s, max(mdp.steps(s,a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in mu.sample_n(num_state_samples)],
            error_tolerance
        )
        
        def deter_policy(state: s) -> A:
            return max(
                ((mdp.step(NonTerminal(state), a).expectation(return_), a)
                 for a in mdp.actions(NonTerminal(state))),
                key=itemgetter(0)
            )[1]
        
        vp.append((this_v, DeterministicPolicy(deter_policy)))
    
    return reversed(vp)
            
def back_opt_qvf(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxQ_Distribution[S, A]],
    gamma: float,
    num_state_samples: int,
    error_tolerance, float
) -> Iterator[QValueFunctionApprox[S, A]]:
    horizon: int = len(mdp_f0_mu_triples)
    qvf: List[QValueFunctionApprox[S, A]] = []
    
    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):
        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            next_return: float = max(
                qvf[i-1]((s1, a)) for a in
                mdp_f0_mu_triples[horizon-1][0].actions(s1)
            ) if i > 0 and isinstance(s1, NonTerminal) else 0.
            return r + gamma * next_return
     
        this_qvf = approx0.solve(
            [((s,a), mdp.step(s,a).expectation(return_))
             for s in mu.sample_n(num_state_samples) for a in mdp.actions(s)],
            error_tolerance
        )
        qvf.append(this_qvf)
    
    return reversed(qvf)


            