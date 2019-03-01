import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
plt.style.use('seaborn-deep')


def acquire_data(data_path='data/stock_prices.csv', n_companies=20):
    """
    Randomly picks 20 companies from the stock csv to act as our selected portfolio
    :param data_path: the path to the csv of stock prices for different tickers
    :param n_companies: the number of companies
    :return: dataframe containing historical price data for 20 tickers
    """
    data = pd.read_csv(data_path, index_col='Date')
    data.dropna(axis=1, how='any', inplace=True)
    cols = np.random.choice(data.shape[1], n_companies, replace=False)
    return data.iloc[:, cols]


def preprocess_data(prices):
    """
    Calculates the annualised mean returns and covariance of price data
    :param prices: dataframe of historical prices for different tickers
    :return: daily returns, annualised mean returns, annualised covariance
    """
    daily_returns = prices.pct_change().dropna()
    return daily_returns, daily_returns.mean() * 252, daily_returns.cov() * 252


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    """
    :param weights: normalised weights
    :param mean_returns: array of mean returns for a number of stocks
    :param cov_matrix: covariance of these stocks.
    :return: the weighted return and weighted volatility of a given portfolio
    """
    mu = weights.dot(mean_returns)
    sigma = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return mu, sigma


def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0):
    """
    :param num_portfolios: the number of random portfolios to generate
    :param mean_returns: array of mean returns for a number of stocks
    :param cov_matrix: covariance of stocks
    :param risk_free_rate: default zero.
    :return: 1. results array holding the portfolio volatility, portfolio return, and Sharpe ratio.
             2. list holding the weights of each portfolio.
    """
    n_assets = len(mean_returns)

    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        # Generate weights and normalise
        w = np.random.random(n_assets)
        w /= np.sum(w)
        weights_record.append(w)
        portfolio_return, portfolio_std_dev = portfolio_annualised_performance(
            w, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

    return results, weights_record


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    """
    The optimisation objective is the negative Sharpe ratio.
    :param weights: normalised weights
    :param mean_returns: array of mean returns for a number of stocks
    :param cov_matrix: covariance of these stocks.
    :param risk_free_rate: defaults to zero
    :return: the negative Sharpe ratio
    """
    portfolio_return, portfolio_std_dev = portfolio_annualised_performance(
        weights, mean_returns, cov_matrix)
    return -(portfolio_return - risk_free_rate) / portfolio_std_dev


def maximise_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0):
    """
    Given the mean returns, covariance matrix and risk free rate, choose weights to maximise
    the Sharpe ratio.
    :param mean_returns: array of mean returns for a number of stocks
    :param cov_matrix: covariance of these stocks.
    :param risk_free_rate: defaults to zero
    :return: the weights that maximise the Sharpe ratio
    """
    n_assets = len(mean_returns)

    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array([1 / n_assets] * n_assets)
    result = sco.minimize(neg_sharpe_ratio, x0=initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def volatility_objective(weights, mean_returns, cov_matrix):
    """
    Volatility of a given portfolio
    :param weights: normalised weights
    :param mean_returns: array of mean returns for a number of stocks
    :param cov_matrix: covariance of these stocks.
    :return:
    """
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]


def minimise_volatility(mean_returns, cov_matrix):
    """
    Given the mean returns, and covariance matrix, choose weights to minimise volatility
    :param mean_returns: array of mean returns for a number of stocks
    :param cov_matrix: covariance of these stocks.
    :return
    """
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for _ in range(n_assets))
    initial_guess = np.array([1 / n_assets] * n_assets)
    result = sco.minimize(volatility_objective, x0=initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def monte_carlo_optimisation(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0):
    """
    This function allows you to compare the monte carlo optimal portfolios with the quadratic programming
    optimal portfolios.
    :param num_portfolios: the number of random portfolios to generate
    :param mean_returns: array of mean returns for a number of stocks
    :param cov_matrix: covariance of stocks
    :param risk_free_rate: default zero.
    :return: Prints the weights for the portfolio with maximum sharpe ratio, and the portfolio with
    minimum variance.
    """
    results, weights = generate_random_portfolios(
        num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    max_sharpe_idx = np.argmax(results[2])

    std_max_sharpe, returns_max_sharpe = results[0,
                                                 max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(
        weights[max_sharpe_idx], index=adj_close.columns, columns=['allocation'])
    max_sharpe_allocation = np.round(max_sharpe_allocation, 3).T

    min_vol_idx = np.argmin(results[0])
    std_min_vol, returns_min_vol = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(
        weights[min_vol_idx], index=adj_close.columns, columns=['allocation'])
    min_vol_allocation = np.round(min_vol_allocation, 3).T

    # Volatility of a stock (variance is found in diagonal of covariance matrix)
    stock_annual_vols = np.sqrt(np.diagonal(cov_matrix))

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(returns_max_sharpe, 3))
    print("Annualised Volatility:", round(std_max_sharpe, 3))
    print(max_sharpe_allocation)

    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(returns_min_vol, 3))
    print("Annualised Volatility:", round(std_min_vol, 3))
    print(min_vol_allocation)

    plt.figure(figsize=(9, 6))
    plt.scatter(results[0, :], results[1, :],
                c=results[2, :], cmap='YlGnBu', marker='o', s=10)
    plt.colorbar()
    plt.scatter(std_max_sharpe, returns_max_sharpe, marker='^',
                color='r', s=100, label="MC Max Sharpe")
    plt.scatter(std_min_vol, returns_min_vol, marker='^',
                color='g', s=100, label='MC Min volatility')
    plt.scatter(stock_annual_vols, mean_returns, marker='o', color='k', s=40)
    for i, ticker in enumerate(adj_close.columns):
        plt.annotate(ticker, (stock_annual_vols[i], mean_returns[i]), xytext=(
            10, 0), textcoords='offset points')

    max_weights = maximise_sharpe_ratio(mean_stock_returns, cov_annual)['x']
    min_weights = minimise_volatility(mean_stock_returns, cov_annual)['x']
    max_sharpe_ret, max_sharpe_var = portfolio_annualised_performance(
        max_weights, mean_stock_returns, cov_annual)
    min_vol_ret, min_vol_var = portfolio_annualised_performance(
        min_weights, mean_stock_returns, cov_annual)
    plt.scatter(max_sharpe_var, max_sharpe_ret, marker='*',
                color='r', s=100, label="QP Max Sharpe")
    plt.scatter(min_vol_var, min_vol_ret, marker='*',
                color='g', s=100, label="QP Min Vol")
    plt.title(
        f"Monte Carlo Optimisation: {num_portfolios} simulated portfolios")
    plt.xlabel("annualised volatility")
    plt.ylabel("annualised returns")
    plt.legend(labelspacing=0.8)
    plt.show()


def efficient_return(mean_returns, cov_matrix, target_return):
    """
    Calculates the minimum risk for a given target return
    :param mean_returns: array of mean returns for a number of stocks
    :param cov_matrix: covariance of these stocks.
    :param target_return: the target return
    :return: the weights of the portfolio that minimise risk for this target return
    """
    n_assets = len(mean_returns)

    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array([1 / n_assets] * n_assets)
    result = sco.minimize(volatility_objective, x0=initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def generate_efficient_frontier(mean_returns, cov_matrix, returns_range):
    """
    :param mean_returns: array of mean returns for a number of stocks
    :param cov_matrix: covariance of these stocks.
    :param returns_range: range of returns for which efficient portfolios will be generated
    :return: a list of weights of efficient portfolios
    """
    efficient_portfolios = []
    for ret in returns_range:
        efficient_portfolios.append(
            efficient_return(mean_returns, cov_matrix, ret))
    return efficient_portfolios


def efficient_frontier_optimisation(mean_returns, cov_matrix, risk_free_rate=0, monte_carlo=None):
    # Maximum sharpe ratio
    max_sharpe = maximise_sharpe_ratio(
        mean_returns, cov_matrix, risk_free_rate)
    returns_max_sharpe, std_max_sharpe = portfolio_annualised_performance(
        max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(
        max_sharpe.x, index=adj_close.columns, columns=['allocation'])
    max_sharpe_allocation = np.round(max_sharpe_allocation, 3).T

    # Minimum volatility
    min_vol = minimise_volatility(mean_returns, cov_matrix)
    returns_min_vol, std_min_vol = portfolio_annualised_performance(
        min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(
        min_vol.x, index=adj_close.columns, columns=['allocation'])
    min_vol_allocation = np.round(min_vol_allocation, 3).T

    # Halfway portfolio
    returns_halfway = (returns_max_sharpe + returns_min_vol) / 2
    halfway = efficient_return(mean_returns, cov_matrix, returns_halfway)
    std_halfway = portfolio_annualised_performance(
        halfway['x'], mean_returns, cov_matrix)[1]
    halfway_allocation = pd.DataFrame(
        halfway.x, index=adj_close.columns, columns=['allocation'])
    halfway_allocation = np.round(halfway_allocation, 3).T

    stock_annual_vols = np.sqrt(np.diagonal(cov_matrix))
    stock_annualised_data = pd.DataFrame({'Annualised returns': mean_returns, 'Annualised vols': stock_annual_vols},
                                         index=adj_close.columns)

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(returns_max_sharpe, 3))
    print("Annualised Volatility:", round(std_max_sharpe, 3))
    print(max_sharpe_allocation)

    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(returns_min_vol, 3))
    print("Annualised Volatility:", round(std_min_vol, 3))
    print(min_vol_allocation)

    print("-" * 80)
    print("Halfway Portfolio Allocation\n")
    print("Annualised Return:", round(returns_halfway, 3))
    print("Annualised Volatility:", round(std_halfway, 3))
    print(halfway_allocation)

    print("-" * 80)
    print("Individual Stock Returns and Volatility\n")
    print(np.round(stock_annualised_data, 3).T)
    print("-" * 80)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(stock_annual_vols, mean_returns, marker='o', s=100)

    if monte_carlo is not None:
        results, weights = generate_random_portfolios(
            monte_carlo, mean_returns, cov_matrix, risk_free_rate)
        ax.scatter(results[0, :], results[1, :], c=results[2, :],
                   cmap='YlGnBu', marker='o', s=10)

    for i, ticker in enumerate(adj_close.columns):
        ax.annotate(ticker, (stock_annual_vols[i], mean_returns[i]), xytext=(
            10, 0), textcoords='offset points')
    ax.scatter(std_max_sharpe, returns_max_sharpe, marker='^',
               color='r', s=200, label='Maximum Sharpe ratio')

    ax.scatter(std_halfway, returns_halfway, marker='^',
               color='darkorchid', s=200, label='Halfway portfolio')

    ax.scatter(std_min_vol, returns_min_vol, marker='^',
               color='g', s=200, label='Minimum volatility')

    target = np.linspace(mean_stock_returns.min(),
                         mean_stock_returns.max(), 50)
    efficient_portfolios = generate_efficient_frontier(
        mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target,
            linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Efficient Frontier Portfolio Optimization')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)
    plt.show()


if __name__ == "__main__":
    adj_close = acquire_data('data/stock_prices.csv')
    daily_stock_returns, mean_stock_returns, cov_annual = preprocess_data(adj_close)

    efficient_frontier_optimisation(
        mean_stock_returns, cov_annual)
