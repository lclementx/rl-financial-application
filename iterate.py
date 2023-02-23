import itertools
from typing import Callable, Iterable, Iterator, Optional, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')

def iterate(step: Callable[[X],X], start: X) -> Iterator[X]:
    state = start
    
    while True:
        yield state
        state = step(state)

def converge(values: Iterator[X], done: Callable[[X,X], bool]) -> Iterator[X]:
    a = next(values, None)
    if a is None:
        return
    
    yield a
    
    for b in values:
        yield b
        if done(a,b):
            return
        a = b
        
#return last value of an iterator. If its iterator is empty then return None      
def last(values: Iterator[X]) -> Optional[X]:
    try:
        *_, last_element = values
        return last_element
    except ValueError:
        return None

#Return the final value of the iterator when it has converged :)
def converged(values: Iterator[X], done: Callable[[X,X], bool]) -> X:
    result = last(converge(values, done))

    if result is None:
        raise ValueError("converged called on an empty iterator")

    return result

def accumulate(
    iterable: Iterable[X],
    func: Callable[[Y, X], Y],
    *,
    initial: Optional[Y]
) -> Iterator[Y]:
    #An iterator that returns accumulated sums, or accumulated results of other 
    #binary functinos (specified via the optional func argument).
    
    if initial is not None:
        iterable = itertools.chain([initial], iterable)
        
    #https://docs.python.org/3/library/itertools.html#itertools.accumulate
    return itertools.accumulate(iterable,func)