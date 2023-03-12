import sys
# setting path
sys.path.append('../') 
import unittest
from wealth_state import *

class TestWealthState(unittest.TestCase):
    ws = WealthState(time=0,wealth=10)
    
    def testWealthStateEquality(self):
        self.assertEqual(self.ws.time,0)
        self.assertEqual(self.ws.wealth,10)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)