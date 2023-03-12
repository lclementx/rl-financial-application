import numpy as np
from wealth_state import WealthState
'''REWARD'''
'''CARA UTILITY'''
def cara_func(x :float, alpha) -> float:
    return - np.exp(-alpha * x)/alpha

def wealth_func(ws: WealthState, risky_asset_allocation, risky_asset_return:float , risk_free_return:float) -> float:
    # print(f'{risky_asset_allocation} * ({risky_asset_return} - {risk_free_return}) + {ws.wealth} * {(1 + risk_free_return)}')
    wealth = risky_asset_allocation * (risky_asset_return - risk_free_return) + ws.wealth * (1 + risk_free_return)
    return wealth