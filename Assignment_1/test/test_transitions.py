import sys
# setting path
sys.path.append('../') 
import unittest
from wealth_state import *
from investment_action import *
from collections import defaultdict
from transitions import *

class TestGetAllStateActions(unittest.TestCase):
    risky_return_dist = {
        0.1: 0.5,
        -0.1: 0.5
    }
    def testGet_All_State_Actions_1_time_step(self):
        ws= WealthState(time=0,wealth=0,termination_time=1)
        actions = {
            InvestmentAction(risky_investment_amount=1):0.5,
            InvestmentAction(risky_investment_amount=0):0.5
        }
        all_state_actions = get_all_state_actions(1,actions,ws,self.risky_return_dist,0)
        expected_state_actions = {
            ws: {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=1,wealth=0.1,termination_time=1):0.25,
                    WealthState(time=1,wealth=-0.1,termination_time=1):0.25
                },
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=1,wealth=0,termination_time=1):0.5 #This ze ro case should've been tested earlier =.=
                }
            }
        }
        self.assertEqual(expected_state_actions,all_state_actions)
        
    def testGet_All_State_Actions_2_time_step(self):
        ws= WealthState(time=0,wealth=0,termination_time=2)
        actions = {
            InvestmentAction(risky_investment_amount=1):0.5,
            InvestmentAction(risky_investment_amount=0):0.5
        }        
        all_state_actions = get_all_state_actions(2,actions,ws,self.risky_return_dist,0)
        expected_state_actions = {
            WealthState(time=0,wealth=0,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=1,wealth=0.1,termination_time=2):0.25,
                    WealthState(time=1,wealth=-0.1,termination_time=2):0.25
                },
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=1,wealth=0,termination_time=2):0.5
                }
            },
            WealthState(time=1,wealth=0.1,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=2,wealth=0.2,termination_time=2):0.25,
                    WealthState(time=2,wealth=0,termination_time=2):0.25
                },
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=2,wealth=0.1,termination_time=2):0.5
                }
            },
            WealthState(time=1,wealth=-0.1,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=2,wealth=0,termination_time=2):0.25,
                    WealthState(time=2,wealth=-0.2,termination_time=2):0.25
                },
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=2,wealth=-0.1,termination_time=2):0.5
                }
            },
            WealthState(time=1,wealth=0,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=2,wealth=0.1,termination_time=2):0.25,
                    WealthState(time=2,wealth=-0.1,termination_time=2):0.25
                },
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=2,wealth=0,termination_time=2):0.5
                }
            }
        }   
        self.assertEqual(expected_state_actions,all_state_actions)

class TestGetStateActionValueMap(unittest.TestCase):
    def testGetStateActionValueMap_1_iteration(self):
        gamma = 1
        probability = 0.5
        state_actions = {
            WealthState(time=0,wealth=0,termination_time=1): {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=1,wealth=0.2,termination_time=1):0.25,
                    WealthState(time=1,wealth=-0.1,termination_time=1):0.25
                },   
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=1,wealth=0,termination_time=1):0.5,
                }
            }
        }
        
        expected_state_action_value_map = defaultdict(float)
        expected_state_action_value_map.update({
            WealthState(time=0,wealth=0,termination_time=1): {
                InvestmentAction(risky_investment_amount=1):0.025,
                InvestmentAction(risky_investment_amount=0):0
        }})
        result = get_state_action_value_map(state_actions,reward_func=lambda x: x,gamma=1,iterations=1)
        self.assertEqual(expected_state_action_value_map,result)
        
    def testGetStateActionValueMap_2_iteration(self):
        state_actions = {
            WealthState(time=0,wealth=0,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=1,wealth=0.1,termination_time=2):0.25,
                    WealthState(time=1,wealth=-0.1,termination_time=2):0.25
                },
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=1,wealth=0,termination_time=2):0.5
                }
            },
            WealthState(time=1,wealth=0.1,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=2,wealth=0.2,termination_time=2):0.25,
                    WealthState(time=2,wealth=0,termination_time=2):0.25
                },
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=2,wealth=0.1,termination_time=2):0.5
                }
            },
            WealthState(time=1,wealth=-0.1,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=2,wealth=0,termination_time=2):0.25,
                    WealthState(time=2,wealth=-0.2,termination_time=2):0.25
                },
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=2,wealth=-0.1,termination_time=2):0.5
                }
            },
            WealthState(time=1,wealth=0,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):{
                    WealthState(time=2,wealth=0.1,termination_time=2):0.25,
                    WealthState(time=2,wealth=-0.1,termination_time=2):0.25
                },
                InvestmentAction(risky_investment_amount=0):{
                    WealthState(time=2,wealth=0,termination_time=2):0.5
                }
            }
        }  
        
        expected_state_action_value_map = defaultdict(float)
        expected_state_action_value_map.update({
            WealthState(time=0,wealth=0,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):0,
                InvestmentAction(risky_investment_amount=0):0
            },
            WealthState(time=1,wealth=0.1,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):0.05,
                InvestmentAction(risky_investment_amount=0):0.05
            },
            WealthState(time=1,wealth=-0.1,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):-0.05,
                InvestmentAction(risky_investment_amount=0):-0.05
            },
            WealthState(time=1,wealth=0,termination_time=2): {
                InvestmentAction(risky_investment_amount=1):0,
                InvestmentAction(risky_investment_amount=0):0
            }
        })
        result = get_state_action_value_map(state_actions,reward_func=lambda x: x,gamma=1,iterations=2)
        self.assertEqual(expected_state_action_value_map,result)
        
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)