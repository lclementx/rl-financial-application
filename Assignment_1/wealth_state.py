from dataclasses import dataclass

'''State'''
@dataclass(frozen=True)
class WealthState:
    time: int #time state I am in
    wealth: float
    termination_time:int = int
    
    def __eq__(self, other):
        return (self.time == other.time) and \
               (self.wealth == other.wealth)
    
    def isTerminal(self):
        return self.time == self.termination_time