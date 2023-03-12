import sys
# setting path
sys.path.append('../') 
import unittest
from utility import *

class TestCARAUtility(unittest.TestCase):
     def testCARA(self):
        # -( e ^ (- alpha * Wt) )/alpha
        self.assertEqual(cara_func(1, alpha=1),-1/np.e)
        self.assertEqual(cara_func(1, alpha=2), - ((np.e ** (-2 * 1))/2))

class TestWealthFunction(unittest.TestCase):
# def wealth_func(ws: WealthState, 
# risky_asset_allocation, 
# risky_asset_return:float , 
# risk_free_return:float = RISK_FREE_RATE) -> float:

    def testWealthFunction_Price_0_0_Risky_Asset(self):
        ws = WealthState(time=0,wealth=1)
        self.assertEqual(wealth_func(ws,0,0,0),1)
        
    def testWealthFunction_Price_1_0_Risky_Asset(self):
        ws = WealthState(time=0,wealth=1)
        self.assertEqual(wealth_func(ws,0,1,0),1)

    def testWealthFunction_Price_0_1_Risky_Asset(self):
        ws = WealthState(time=0,wealth=1)
        self.assertEqual(wealth_func(ws,1,0,0),1)
        
    def testWealthFunction_Price_2_1_Risky_Asset(self):
        ws = WealthState(time=0,wealth=1)
        self.assertEqual(wealth_func(ws,1,2,0),3)
        
    def testWealthFunction_Price_neg_3_2_Risky_Asset(self):
        ws = WealthState(time=0,wealth=1)
        self.assertEqual(wealth_func(ws,2,-3,0),-5)

    def testWealthFunction_Price_neg_3_2_Risky_Asset_1bps_Risk_Free_Rate(self):
        ws = WealthState(time=0,wealth=1)
        self.assertEqual(wealth_func(ws,2,-3,0.01),-5.01)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)