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
from dynamic_programming import *

A = TypeVar('A')
S = TypeVar('S')

V = Mapping[NonTerminal[S], float]

RewardOutcome = FiniteDistribution[Tuple[WithTime[S], float]]

#Finite Horizon Markov Reward Process
#Turn a normal FiniteMarkovRewardProcess into one with a finite horizon that stops after 'limit' steps.

def finite_horizon_MRP(
    process: FiniteMarkovRewardProcess[S],
    limit: int
) -> FiniteMarkovRewardProcess[WithTime[S]]:
    transition_map: Dict[WithTime[S], RewardOutcome] = {}
    
    #Non-terminal states
    for time in range(limit):
        for s in process.non_terminal_states:
            result: StateReward[S] = process.transition_reward(s)
            s_time = WithTime(state=s.state, time=time)
            
            transition_map[s_time] = result.map(
                lambda sr: (WithTime(state=sr[0].state, time=time+1), sr[1])
            )
    
    return FiniteMarkovRewardProcess(transition_map)
    

def unwrap_finite_horizon_MRP(
    process: FiniteMarkovRewardProcess[WithTime[S]]
) -> Sequence[RewardTransition[S]]:
    def time(x: WithTime[S]) -> int:
        return x.time
    
    def single_without_time(
        s_r: Tuple[State[WithTime[S]], float]
    ) -> Tuple[State[S], float]:
        #Remember this is a State Reward Mapping so s_r[0] = state
        if isinstance(s_r[0], NonTerminal):
            ret: Tuple[State[S], float] = (
                NonTerminal(s_r[0].state.state),
                s_r[1]
            )
        else:
            ret = (Terminal(s_r[0].state.state),s_r[1])
        return ret
    
    #wrap all states with Terminal or NonTerminal, strips time away from state.
    def without_time(arg: StateReward[WithTime[S]]) -> StateReward[S]:
        return arg.map(single_without_time)
    
    return [{NonTerminal(s.state) : without_time( #[{State_0: {State_3:1, State:4,5}, State_1: {State_5:9}] <- in time sequence
        #Transition_reward returns Distribution[Tuple[State[S], float]]
        process.transition_reward(NonTerminal(s))
    ) for s in states} for _, states in groupby( #Groupby -> {t=0: [State_0,State_1,State_2], t=1:[State_3,State_4,State_5]}
        sorted(
            (nt.state for nt in process.non_terminal_states),
            key=time
        ),
        key=time
    )]

def finite_horizon_MDP(
    process: FiniteMarkovDecisionProcess[S, A],
    limit: int
) -> FiniteMarkovDecisionProcess[WithTime[S], A]:
    mapping: Dict[WithTime[S], Dict[A, FiniteDistribution[Tuple[WithTime[S],float]]]]= {}
    #Non-terminal states
    for time in range(0, limit):
        for s in process.non_terminal_states:
            s_time = WithTime(state=s.state, time=time)
            mapping[s_time] = {a: result.map(
                lambda sr: (WithTime(state=sr[0].state, time=time+1), sr[1])
            ) for a, result in process.mapping[s].items()}
            
    return FiniteMarkovDecisionProcess(mapping)

def unwrap_finite_horizon_MDP(
    process: FiniteMarkovDecisionProcess[WithTime[S], A]
) -> Sequence[StateActionMapping[S, A]]:
    def time(x: WithTime[S]) -> int:
        return x.time
    
    def single_without_time(
        s_r: Tuple[State[WithTime[S]], float]
    ) -> Tuple[State[S], float]:
        #Remember this is a State Reward Mapping so s_r[0] = state
        if isinstance(s_r[0], NonTerminal):
            ret: Tuple[State[S], float] = (
                NonTerminal(s_r[0].state.state),
                s_r[1]
            )
        else:
            ret = (Terminal(s_r[0].state.state),s_r[1])
        return ret
    
    #wrap all states with Terminal or NonTerminal, strips time away from state.
    def without_time(arg: ActionMapping[A, WithTime[S]]) -> ActionMapping[A, S]:
        return {a: sr_distr.map(single_without_time) for a, sr_distr in arg.items()}
    
    return [{NonTerminal(s.state) : without_time(
        process.mapping[NonTerminal(s)]
    ) for s in states} for _, states in groupby( #Groupby -> {t=0: [State_0,State_1,State_2], t=1:[State_3,State_4,State_5]}
        sorted(
            (nt.state for nt in process.non_terminal_states),
            key=time
        ),
        key=time
    )]

#This does backwards induction :) Given a sequence of reward transitions
def evaluate(
    steps: Sequence[RewardTransition[S]],
    gamma: float
) -> Iterator[V[S]]:
    v: List[V[S]] = []
    for step in reversed(steps):
        v.append({s: res.expectation(
            lambda s_r: s_r[1] + gamma * (
                extended_vf(v[-1], s_r[0]) if len(v) > 0 else 0.
            )
        ) for s, res in step.items()})
    return reversed(v)

def optimal_vf_and_policy(
    steps: Sequence[StateActionMapping[S, A]],
    gamma: float
) -> Iterator[Tuple[V[S], FiniteDeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step
    '''
    v_p: List[Tuple[V[S], FiniteDeterministicPolicy[S, A]]] = []

    for step in reversed(steps):
        this_v: Dict[NonTerminal[S], float] = {}
        this_a: Dict[S, A] = {}
        for s, actions_map in step.items():
            action_values = ((res.expectation(
                lambda s_r: s_r[1] + gamma * (
                    extended_vf(v_p[-1][0], s_r[0]) if len(v_p) > 0 else 0.
                )
            ), a) for a, res in actions_map.items())
            v_star, a_star = max(action_values, key=operator.itemgetter(0))
            this_v[s] = v_star
            this_a[s.state] = a_star
        v_p.append((this_v, FiniteDeterministicPolicy(this_a)))

    return reversed(v_p)