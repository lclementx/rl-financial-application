import sys
# setting path
sys.path.append('../') 
import unittest
from investment_action import *

def assert_lists_no_order(test_case, list1,list2):
    for i in list1:
        test_case.assertIn(i,list2)
        
    for i in list2:
        test_case.assertIn(i,list1)
        
class TestInvestmentAction(unittest.TestCase):
    ia = InvestmentAction(risky_investment_amount=1)
    ws = WealthState(time=0,wealth=10)
    
    def testNextState(self):
        ns = self.ia.action_to_next_state(self.ws,0.2,0.01)
        self.assertEqual(ns.time,1)
        self.assertEqual(ns.wealth,10.29)

    def testGetAllActions_Investment_Limit_1(self):
        ia = InvestmentAction(risky_investment_amount=1)
        actions = InvestmentAction.get_all_actions(1,-1,3)
        expected_actions = [
            InvestmentAction(risky_investment_amount=1),
            InvestmentAction(risky_investment_amount=0),
            InvestmentAction(risky_investment_amount=-1)
        ]
        assert_lists_no_order(self,actions,expected_actions)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)