import numpy as np
from scipy.stats import norm

# Parameters
initial_premium = 10000  # Initial premium
guarantee_level = 10000  # Guarantee level (100% of premium)
volatility = 0.1449  # Volatility of S&P 500 for the last 10 years
risk_free_rate = 0.045  # Risk-free interest rate 
mortality_rates = [
    0.0099, 0.01105, 0.0122, 0.01335, 0.0145, 0.01565, 0.0168, 0.01795, 0.0191, 0.02025,
    0.0214, 0.02255, 0.0237, 0.02485, 0.026, 0.02715, 0.0283, 0.02945, 0.0306, 0.03175,
    0.0329, 0.03405, 0.0352, 0.03635, 0.0375, 0.03865, 0.0398, 0.04095, 0.0421, 0.04325,
    0.0444, 0.04555, 0.0467, 0.04785, 0.049
]  # Sample mortality rates for ages 65-99
T = 35  # Time to maturity in years
n_simulations = 1000  # Number of Monte Carlo simulations

# Black-Scholes formula for European put option
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Simulate GBM paths
def simulate_gbm_paths(S_0, mu, sigma, T, n_simulations):
    dt = 1  # Time step in years
    paths = np.zeros((n_simulations, T + 1))
    paths[:, 0] = S_0
    for t in range(1, T + 1):
        Z = np.random.standard_normal(n_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

# Calculate the present value of GMDB cost at time 0
def calculate_gmdb_price(initial_premium, guarantee_level, T, r, sigma, mortality_rates):
    gmdb_price = 0
    survival_probability = 1

    # Simulate GBM paths for account value
    account_paths = simulate_gbm_paths(initial_premium, risk_free_rate, volatility, T, n_simulations)

    for t in range(1, T + 1):
        # Calculate average account value at time t
        account_value_t = np.mean(account_paths[:, t])

        # Calculate put option price
        put_price = black_scholes_put(S=account_value_t, K=guarantee_level, T=T - t + 1, r=r, sigma=sigma)

        # Calculate the present value of the GMDB for this year
        gmdb_curr = put_price * survival_probability * mortality_rates[t - 1] * np.exp(-r * t)
        gmdb_price += gmdb_curr

        # Update survival probability
        survival_probability *= (1 - mortality_rates[t - 1])

        # Reset mechanism
        if t <= 15:  # Ensure reset stops at age 80 (15 years from age 65)
            guarantee_level = max(guarantee_level, account_value_t)

        print(f"Year: {t}, GMDB Current: {gmdb_curr:.2f}, Account Value: {account_value_t:.2f}, Guarantee Level: {guarantee_level:.2f}")

    return gmdb_price

# Present value of fees function
def present_value_of_fees(fee_percentage, initial_premium, r, T):
    present_value = 0
    account_value = initial_premium
    for t in range(1, T + 1):
        annual_fee = account_value * fee_percentage
        present_value += annual_fee * np.exp(-r * t)  # Discount the fee to present values
        account_value *= (1 + r)  # Update account value for next year

    return present_value

# Find best fee function
def find_best_fee(gmdb_price):
    fee_percentage = 0.001  # Start with a guess (0.1%)
    step = 0.0000001  # Step size for adjusting the guess
    tolerance = 0.1  # Tolerance for the difference in cost

    while True:
        pv_fees = present_value_of_fees(fee_percentage, initial_premium, risk_free_rate, T)
        if abs(pv_fees - gmdb_price) < tolerance:
            break
        elif pv_fees < gmdb_price:
            fee_percentage += step
        else:
            fee_percentage -= step

    return fee_percentage

# Calculate the GMDB price for the given parameters
gmdb_price = calculate_gmdb_price(initial_premium, guarantee_level, T, risk_free_rate, volatility, mortality_rates)
print(f"The estimated price of the GMDB over 35 years is: {gmdb_price:.2f}")

# Find the best fee percentage
best_fee = find_best_fee(gmdb_price)
print(f"The best annual fee as a percentage is: {best_fee * 100:.4f}%")

# Additional scenarios
'''
print("Initial premium \t Guarantee \t Risk-free rate \t Volatility \t GMDB")
for i in [5000, 10000, 50000]:
    g = i
    for r in [0.03, 0.05, 0.1]:
        for v in [0.15, 0.2, 0.25]:
            gmdb_price = calculate_gmdb_price(i, g, T, r, v, mortality_rates)
            print(f"{i}\t{g}\t{r}\t{v}\t{gmdb_price:.2f}")
'''
