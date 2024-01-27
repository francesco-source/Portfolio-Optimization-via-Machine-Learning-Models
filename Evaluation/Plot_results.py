import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def plot_portfolio_cumulative_return(portfolio_df, 
                                     figsize= (16, 6),
                                     title= 'Unsupervised Learning Trading Strategy Returns Over Time',
                                     refering_index= None,
                                     random_returns_df= None,
                                     save_path= None):
    """
    Plot cumulative return of a portfolio over time.

    Args:
    - portfolio_df (pd.DataFrame): DataFrame containing portfolio returns.
    - figsize (tuple, optional): Size of the figure. Defaults to (16, 6).
    - title (str, optional): Title of the plot. Defaults to 'Unsupervised Learning Trading Strategy Returns Over Time'.
    - random (pd.DataFrame, optional): The cumulative returns using a random baseline with confidence intervals.

    Returns:
    - None
    """
    # Calculate cumulative return
    portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1
    
    # Plot
    plt.figure(figsize=figsize)

    color_palette = ('r', 'g', 'c', 'm')
    
    for i, strategy in enumerate(portfolio_cumulative_return.columns):
        portfolio_cumulative_return[strategy].plot(ax=plt.gca(), color= color_palette[i])

    if refering_index is not None:
        index_cum_return = np.exp(np.log1p(refering_index["return"]).cumsum()) - 1
        plt.plot(index_cum_return.index, index_cum_return.values, label= refering_index["name"], linestyle= 'dashed', color= 'blue')

    if random_returns_df is not None:
        plt.plot(random_returns_df["Mean"].index, random_returns_df["Mean"].values, color='gold', label= 'Random baseline')
        plt.fill_between(random_returns_df["Mean"].index, 
                         random_returns_df["Down"].values, 
                         random_returns_df["Up"].values, 
                         color='gold',
                         alpha= .2)
        plt.legend()

    plt.title(title)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel('Return')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

# Example usage:
# plot_portfolio_cumulative_return(portfolio_df_6)
