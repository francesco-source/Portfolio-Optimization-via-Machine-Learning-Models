import numpy as np

def calculate_volatility(returns):
    """
    Calculate the volatility of returns.

    Args:
    - returns (array-like): Array-like object containing the returns data.

    Returns:
    - float: Volatility of the returns.
    """
    volatility = np.std(returns)
    return volatility

def calculate_sharpe_ratio(returns,
                           risk_free_rate=0):
    """
    Calculate the Sharpe ratio.

    The Sharpe ratio is a measure of risk-adjusted return, 
    representing the ratio of the excess return of an investment
    (over the risk-free rate) to its volatility.

    Args:
    - returns (array-like): Array-like object containing the returns data.
    - risk_free_rate (float, optional): Risk-free rate. Defaults to 0.

    Returns:
    - float: Sharpe ratio.
    """
    avg_return = np.mean(returns)
    volatility = calculate_volatility(returns)
    sharpe_ratio = (avg_return - risk_free_rate) / volatility
    return sharpe_ratio