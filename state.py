from __future__ import annotations
from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass
from typing import Generic, Callable, TypeVar, Iterable

S = TypeVar('S')
X = TypeVar('X')

#Define what a state is. You have Terminal States and Non Terminal States
class State(ABC, Generic[S]):
    state: S
    
    def on_non_terminal(
        self,
        f: Callable[[NonTerminal[S]], X],
        default: X
    ) -> X:
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default

@dataclass(frozen=True)
class Terminal(State[S]):
    state: S

@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S
    
    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.state < other.state

#State including time as index to keep it markov
@dataclass(frozen=True)
class WithTime(Generic[S]):
    state: S
    time: int = 0
    
    def step_time(self) -> WithTime[S]:
        return dataclasses.replace(self, time=self.time + 1)