from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Sequence
from wealth_state import WealthState
from utility import *

'''ACTIONS'''
@dataclass(frozen=True)
class InvestmentAction:
    risky_investment_amount: float
    
    def __eq__(self, other):
        return self.risky_investment_amount == other.risky_investment_amount
    
    def action_to_next_state(self, ws: WealthState, risky_return, risk_free_rate) -> WealthState:
        next_expected_wealth = wealth_func(ws, self.risky_investment_amount, risky_return, risk_free_rate)
        new_state = WealthState(time=ws.time+1, 
                                wealth=next_expected_wealth,
                                termination_time=ws.termination_time
                               )
        return new_state
    
    @staticmethod
    def get_all_actions(investment_limit_min: int,
                        invesment_limit_max: int,
                        splits: int
                       ) -> Sequence[InvestmentAction]:
        all_actions = list()
        allocations = np.linspace(investment_limit_min,invesment_limit_max,splits).tolist()
        for alloc in allocations:
            all_actions.append(InvestmentAction(risky_investment_amount=alloc))
        
        return all_actions