import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import discrete_allocation

# Read in price data
df = pd.read_hdf("stock_prices.csv", "df")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

latest_prices = discrete_allocation.get_latest_prices(df)
allocation, leftover = discrete_allocation.portfolio(
    weights, latest_prices, total_portfolio_value=10000
)
print(allocation)
print("Funds remaining: ${:.2f}".format(leftover))
