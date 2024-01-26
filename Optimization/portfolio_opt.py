from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def optimize_weights(prices, lower_bound=0):
    """
    Optimize portfolio weights to maximize the Sharpe ratio.

    Args:
        prices (pd.DataFrame): DataFrame containing historical prices of assets.
        lower_bound (float, optional): Lower bound for asset weights. Defaults to 0.

    Returns:
        dict: Optimized portfolio weights maximizing the Sharpe ratio.
    """
    # Calculate historical expected returns
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
    
    # Estimate covariance matrix of asset returns
    cov = risk_models.sample_cov(prices=prices, frequency=252)
    
    # Initialize EfficientFrontier object with expected returns and covariance matrix
    ef = EfficientFrontier(expected_returns=returns, 
                           cov_matrix=cov, 
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')
    
    # Optimize portfolio weights to maximize the Sharpe ratio
    weights = ef.max_sharpe()
    
    return weights



def buy_or_sell(dataframe, start_date,
                port_weights,
                choosen_stocks,
                off_set_days=30,
                mode="Regression",
                order=(6, 1, 1)):
    """
    Determine whether to buy or sell stocks based on angular coefficients of past data.

    Args:
        dataframe (pd.DataFrame): DataFrame containing stock data.
        start_date (str): Start date for analyzing past data.
        port_weights (pd.DataFrame): DataFrame containing portfolio weights.
        choosen_stocks (list): List of chosen stock symbols.
        off_set_days (int, optional): Number of days to offset from start_date. Defaults to 30.
        mode (str, optional): Mode for calculating angular coefficients ("Regression" or "Arima").
                                Defaults to "Regression".
        order (tuple, optional): Order of the ARIMA model if mode is "Arima". Defaults to (6, 1, 1).

    Returns:
        float: Total returns based on the calculated angular coefficients.
    """
    # Extract closing prices from dataframe
    closing_prices = dataframe["Adj Close"]
    
    # Convert start_date to timestamp and calculate end_date with offset
    start_date_timestamp = pd.to_datetime(start_date)
    end_date = start_date_timestamp - pd.offsets.Day(off_set_days)
    
    # Extract past data for chosen stocks within the specified timeframe
    past_data = closing_prices.loc[end_date:start_date_timestamp, choosen_stocks]
    
    # Initialize DataFrame to store angular coefficients
    angular_coeff_df = pd.DataFrame(index=choosen_stocks, columns=['Angular_Coefficient'])
    
    # Calculate angular coefficients based on selected mode
    if mode == "Regression":
        for stock in choosen_stocks:
            try:
                # Fit linear regression model
                linear_reg = LinearRegression()
                x_values = np.arange(len(past_data))[:, np.newaxis]
                linear_reg.fit(x_values, past_data[stock])
                slope = linear_reg.coef_[0]
                angular_coeff_df.loc[stock, 'Angular_Coefficient'] = slope
            except:
                angular_coeff_df.loc[stock, 'Angular_Coefficient'] = 0

    elif mode == "Arima":
        for stock in choosen_stocks:
            try:
                # Fit ARIMA model
                arima = ARIMA(past_data[stock], order=order)
                result = arima.fit()
                
                # Generate forecast
                forecast_steps = 5
                forecast = result.get_forecast(steps=forecast_steps)
                forecast = pd.Series(forecast.predicted_mean)
                
                # Fit linear regression model to forecast
                linear_reg = LinearRegression()
                x_values = np.arange(len(forecast))[:, np.newaxis]
                linear_reg.fit(x_values, forecast)
                slope = linear_reg.coef_[0]
                angular_coeff_df.loc[stock, 'Angular_Coefficient'] = slope
            except:
                angular_coeff_df.loc[stock, 'Angular_Coefficient'] = 0

    # Calculate returns based on angular coefficients and portfolio weights
    returns = []
    for i in range(len(angular_coeff_df)):
        returns.append(angular_coeff_df.iloc[i, 0] * port_weights.T.iloc[i, 0])
    
    # Return total returns
    return np.sum(returns)


def daily_portfolio_return(choosen_stocks: dict,
                           fresh_data: pd.DataFrame,
                           stop_loss=False, off_set_days=10,
                           mode="Regression",
                           order = (6,1,1)):
    """
    Calculate daily portfolio returns based on selected stocks and historical price data.

    Args:
        choosen_stocks (dict): Dictionary containing chosen stocks for each start date.
        fresh_data (pd.DataFrame): DataFrame containing fresh historical price data of assets.
        stop_loss (bool, optional): Flag to enable stop-loss mechanism. Defaults to False.
        off_set_days (int, optional): Number of days to offset from start_date. Defaults to 10.
        mode (str, optional): Mode for calculating angular coefficients ("Regression" or "Arima"). 
                              Defaults to "Regression".

    Returns:
        pd.DataFrame: DataFrame containing daily portfolio returns.
    """
    counter = 0
    returns_dataframe = np.log(fresh_data['Adj Close']).diff()  # Compute daily returns
    portfolio_df = pd.DataFrame()  # Initialize DataFrame to store portfolio returns

    for start_date in choosen_stocks.keys():  # Iterate over start dates of each month
        try:
            end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')  # End date of the month
            cols = choosen_stocks[start_date]  # Stocks chosen for the month

            # Define start and end dates for the optimization window (one year prior to the start date)
            optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
            optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')

            # Historical prices for optimization
            optimization_df = fresh_data[optimization_start_date:optimization_end_date]['Adj Close'][cols]  

            success = False
            try:
                # Optimize portfolio weights using historical prices
                weights = optimize_weights(prices=optimization_df, lower_bound=0.001)
                weights = pd.DataFrame(weights, index=pd.Series(0))
                success = True
            except:
                pass
            
            if success == False:
                # If optimization fails, use equal weights for all stocks
                weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                       index=optimization_df.columns.tolist(), columns=pd.Series(0)).T
            
            if stop_loss:
                # Apply stop-loss mechanism
                slope_returns = buy_or_sell(fresh_data, start_date=start_date, port_weights=weights,
                                            choosen_stocks=cols, off_set_days=off_set_days, mode=mode,order = order)

            temp_df = returns_dataframe[start_date:end_date]  # Daily returns for the month
            temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True), left_index=True,
                       right_index=True).reset_index().set_index(['Date', 'index']).unstack().stack()
            temp_df.index.names = ['date', 'ticker']
            temp_df['weighted_return'] = temp_df['return'] * temp_df['weight']  # Compute weighted returns
            temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')  # Aggregate daily returns
            
            if stop_loss and slope_returns < 0.:
                # If stop-loss is triggered, set portfolio return to 0
                temp_df["Strategy Return"] = 0.
                counter += 1

            portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)  # Concatenate monthly returns to portfolio DataFrame

        except Exception as e:
            pass
    
    if stop_loss:
        # Display number of times the stop-loss filter was triggered
        print("Number of times the filter has been used: ", counter)

    return portfolio_df  # Return DataFrame containing daily portfolio returns

