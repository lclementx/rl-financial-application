'''optimal_step_analytical'''
import matplotlib.pyplot as plt
import numpy as np

def analytical_solution(x, ITERATIONS, PROBABILITY_PRICE_UP, PRICE_A, PRICE_B, RISK_FREE_RATE, COEFFICIENT_OF_CARA):
    probability_part = np.log((PROBABILITY_PRICE_UP * (RISK_FREE_RATE - PRICE_A)) / ((PROBABILITY_PRICE_UP-1) * (RISK_FREE_RATE - PRICE_B)))
    time_part = COEFFICIENT_OF_CARA * ( (1 + RISK_FREE_RATE) ** (ITERATIONS - x - 1) ) * (PRICE_A - PRICE_B)
    optimal_step = (1/time_part) * probability_part
    return(optimal_step)

def expected_utility(x, WEALTH, PROBABILITY_PRICE_UP, PRICE_A, PRICE_B, RISK_FREE_RATE, COEFFICIENT_OF_CARA):
    price_a_wealth = PROBABILITY_PRICE_UP * np.exp(-COEFFICIENT_OF_CARA * (x * (PRICE_A - RISK_FREE_RATE)))
    price_b_wealth = (1-PROBABILITY_PRICE_UP) * np.exp(-COEFFICIENT_OF_CARA * (x * (PRICE_B - RISK_FREE_RATE)))
    risk_free_wealth = np.exp(-COEFFICIENT_OF_CARA * WEALTH * (1 + RISK_FREE_RATE))
    return - (risk_free_wealth * (price_a_wealth + price_b_wealth))/COEFFICIENT_OF_CARA

def plot_analytical_solution(
    ITERATIONS, 
    INITIAL_WEALTH, 
    PROBABILITY_PRICE_UP, 
    PRICE_A, 
    PRICE_B, 
    RISK_FREE_RATE, 
    COEFFICIENT_OF_CARA
):
    time_step = [ _ for _ in range(ITERATIONS) ]
    wealth = INITIAL_WEALTH
    risky_asset_return_dist = {
        PRICE_A:PROBABILITY_PRICE_UP,
        PRICE_B:1-PROBABILITY_PRICE_UP
    }
    x = np.linspace(-10, 10, 200)
    fig, axs = plt.subplots(len(time_step),figsize=(20, 5 * len(time_step)))
    for time in time_step:
        step = time
        optimal_alloc = analytical_solution(step,ITERATIONS, PROBABILITY_PRICE_UP, PRICE_A, PRICE_B, RISK_FREE_RATE, COEFFICIENT_OF_CARA)
        expected_wealth = 0
        for returns, prob in risky_asset_return_dist.items():
            expected_wealth += prob * (optimal_alloc * (returns - RISK_FREE_RATE) + wealth * (1 + RISK_FREE_RATE))

        wealth=expected_wealth
        y = [expected_utility(c,wealth,PROBABILITY_PRICE_UP, PRICE_A, PRICE_B, RISK_FREE_RATE, COEFFICIENT_OF_CARA) for c in x ]
        y_alloc = expected_utility(optimal_alloc,wealth,PROBABILITY_PRICE_UP, PRICE_A, PRICE_B, RISK_FREE_RATE, COEFFICIENT_OF_CARA)
        axs[time].plot(x,y)
        axs[time].scatter([optimal_alloc],[y_alloc],color='r',s=50,marker='x')
        axs[time].text(optimal_alloc, y_alloc, '({}, {})'.format(optimal_alloc, y_alloc),fontsize
    ='large')
        axs[time].set_title(f'Time Step: {time}')
        axs[time].set_xlabel('alloc')
        axs[time].set_ylabel('utility')
        print(f'Time: {time} - Optimal Allocation: {optimal_alloc}, Expected Utility: {y_alloc}')

    plt.show()