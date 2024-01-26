import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def plot_portfolio_cumulative_return(portfolio_df, 
                                     figsize=(16, 6),
                                     title='Unsupervised Learning Trading Strategy Returns Over Time'):
    """
    Plot cumulative return of a portfolio over time.

    Args:
    - portfolio_df (pd.DataFrame): DataFrame containing portfolio returns.
    - figsize (tuple, optional): Size of the figure. Defaults to (16, 6).
    - title (str, optional): Title of the plot. Defaults to 'Unsupervised Learning Trading Strategy Returns Over Time'.

    Returns:
    - None
    """
    # Calculate cumulative return
    portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1
    
    # Plot
    plt.figure(figsize=figsize)
    portfolio_cumulative_return.plot(ax=plt.gca())
    plt.title(title)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel('Return')
    plt.show()

# Example usage:
# plot_portfolio_cumulative_return(portfolio_df_6)
