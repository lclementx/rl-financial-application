from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Callable, TypeVar, Iterable,\
Optional, Mapping, Tuple
from collections import defaultdict
from distribution import Distribution, FiniteDistribution, Constant
from state import State, NonTerminal

A = TypeVar('A')
S = TypeVar('S')

#Define a policy - given a state, return a distribution for every action possible for a given state
class Policy(ABC, Generic[S, A]):
    @abstractmethod
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        pass

@dataclass(frozen=True)
class DeterministicPolicy(Policy[S, A]):
    action_for: Callable[[S], A]
    
    #Action with probability 1
    def act(self, state: NonTerminal[S]) -> Constant[A]:
        return Constant(self.action_for(state.state))

#Define a finite policy that maps each non-terminal state to a probability distribution over a finite set of actions
@dataclass(frozen=True)
class FinitePolicy(Policy[S, A]):
    policy_map: Mapping[S, FiniteDistribution[A]]
    
    def __repr__(self) -> str:
        display=""
        for s, d in self.policy_map.items():
            display += f"For State {s}:\n"
            for a, p in d:
                display += f"  Do Action {a} with Probability {p: .3f}\n"
            return display
    
    def act(self, state: NonTerminal[S]) -> FiniteDistribution[A]:
        return self.policy_map[state.state]

#Define a DETERMINISTIC policy i.e. the single action has a probability of 1 happening. 
#(which is the Constant distribution defined)
class FiniteDeterministicPolicy(FinitePolicy[S, A]):
    action_for: Mapping[S, A]
    
    def __init__(self, action_for: Mapping[S, A]):
        self.action_for = action_for
        super().__init__(policy_map={s: Constant(a) for s, a in
                                     self.action_for.items()})
    
    def __repr__(self) -> str:
        display = ""
        for s, a in self.action_for.items():
            display += f"For State {s}: Do Action {a}\n"
        return display