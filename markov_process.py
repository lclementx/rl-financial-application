from __future__ import annotations
import numpy as np
import itertools
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Callable, TypeVar, Iterable,\
Optional, Mapping, Tuple
from collections import defaultdict
from distribution import Distribution, Categorical, SampledDistribution, Constant, \
FiniteDistribution
from state import *

S = TypeVar('S')
X = TypeVar('X')

#Outline a Markov Process
#The state_state_distribution is a probability distribution of Non Terminal States. To start, you sample
#from the distribution and pick a random start state. --> then transition!
class MarkovProcess(ABC, Generic[S]):
    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        pass
    
    #Simulate - transition state until Terminal state
    def simulate(
        self,
        start_state_distribution: Distribution[NonTerminal[S]]
    )-> Iterable[State[S]]:
        
        state: State[S] = start_state_distribution.sample()
        yield state
        
        while isinstance(state, NonTerminal):
            state = self.transition(state).sample()
            yield state

#A finite markov process - extension of a Markov Process that will take in a transition map (Finite Transitions)
#its mapping from State --> Probability Distribtuion of next state. This way you can retrieve the probability distribution
#and sample it to get the next state (transition)
class FiniteMarkovProcess(MarkovProcess[S]):
    non_terminal_states: Sequence[NonTerminal[S]]
    transition_map: Transition[S]
    
    def __init__(self, transition_map: Mapping[S, FiniteDistribution[S]]):
        non_terminals: Set[S] = set(transition_map.keys())
        self.transition_map = {
            NonTerminal(s): Categorical(
                {(NonTerminal(s1) if s1 in non_terminals else Terminal(s1)) : p 
                  for s1, p in v }
            ) for s, v in transition_map.items()
        }
        self.non_terminal_states = list(self.transition_map.keys())
    
    #Fancy printing for visualizing the transition map
    def __repr__(self) -> str:
        display = ""
        
        for s, d in self.transition_map.items():
            display += f"From State {s.state}: \n"
            for s1, p in d:
                opt = (
                    "Terminal State" if isinstance(s1, Terminal) else "State"
                )
                display += f" To {opt} {s1.state} with Probability {p:.3f}\n"
        return display
    
    def transition(self, state: NonTerminal[S]) -> FiniteDistribution[State[S]]:
        return self.transition_map[state]
    
    #Transition matrix - put all the transition probabilities into a matrix (all non-terminal states)
    def get_transition_matrix(self) -> np.ndarray:
        sz = len(self.non_terminal_states) #size
        mat = np.zeros((sz,sz))
        
        for i, s1 in enumerate(self.non_terminal_states):
            for j, s2 in enumerate(self.non_terminal_states):
                mat[i,j] = self.transition(s1).probability(s2)
        
        return mat
    
    #Using the transition matrix, calcualte the stationary distribution - this the probability distribution
    #that remains unchanged as time progresses. Typically it is represented by a row vector pi the probabilities
    #are summed to 1. Given transition matrix P, it satisfies pi = pi * P i.e. it is invariant by P.
    def get_stationary_distribution(self) -> FiniteDistribution[S]:
        eig_vals , eig_vecs = np.linalg.eig(self.get_transition_matrix().T)
        index_of_first_unit_eig_val = np.where(
            np.abs(eig_vals - 1) < 1e-8)[0][0]
        eig_vec_of_unit_eig_val = np.real(
            eig_vecs[:, index_of_first_unit_eig_val])
        return Categorical({
            self.non_terminal_states[i].state: ev
            for i, ev in enumerate(eig_vec_of_unit_eig_val / sum(eig_vec_of_unit_eig_val))
        })