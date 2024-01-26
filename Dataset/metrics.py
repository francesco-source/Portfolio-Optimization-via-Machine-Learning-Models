import numpy as np
import pandas as pd
from typing import Union
import pandas_ta
import pandas_datareader.data as web
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm


def garman_klass_volatility(df: pd.DataFrame):
    """  
    Volatility measure that takes into account also highs
    and lows inside each timestamp.
  
    Args:
        df (pd.DataFrame): daily dataframe of yfinance format.

    Returns:
        GKV : numpy array containing the volatility of each day.
    """
    return ((np.log(df['high']) - np.log(df['low']))**2) / 2 - \
                        (2*np.log(2) - 1) * ((np.log(df['adj close']) - np.log(df['open']))**2)


def rsi(df: pd.DataFrame, length: int = 20):
    """
        It gauges the velocity and extent of recent
        price changes in an asset, providing insights
        into whether the stock is currently in a bullish
        or bearish period.

    Args:
        df (pd.DataFrame): daily dataframe of yfinance format.
        length (int, optional): window choosed. Defaults to 20.

    Returns:
        RSI : numpy array containing the index of each day.
    """
    return df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close= x, length= length))


def bollinger_band(df: pd.DataFrame, length: int = 20, index= 1):
    """ 
    Calculate the bolliger band of the desired index. 
        LOW = 0
        MID = 1
        HIGH = 2

    Args:
        df (pd.DataFrame): daily dataframe of yfinance format.
        length (int, optional): window choosed. Defaults to 20.
        index (int, optional): which bollinger band. Defaults to 1.

    Returns:
        BBand : numpy array containing the index of each day.
    """
    return df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x),
                                                                                 length= length).iloc[:, index])


def norm_atr(df: pd.DataFrame):
    """
     ATR considers not only the volatility of that day
     but also takes into account the one of the previous 
     14 days usually

    Args:
        df (pd.DataFrame): daily dataframe of yfinance format.
    
    Returns:
        ATR : numpy array containing the index of each day.
    """
    def compute_atr(stock_data):
        atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
        return atr.sub(atr.mean()).div(atr.std())
    
    return df.groupby(level=1, group_keys=False).apply(compute_atr)


def macd(df: pd.DataFrame):
    """
    It represents the disparity between
    two moving averages: one calculated over a
    12-day period and the other over 26 days. If it
    is positive it means we are on a bullish period,
    while if it is negative in a bearish one. 

    Args:
        df (pd.DataFrame): daily dataframe of yfinance format.
    
    Returns:
        MACD : numpy array containing the index of each day.
    """
    def compute_macd(close):
        macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
        return macd.sub(macd.mean()).div(macd.std())

    return df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)


def dollar_volume(df: pd.DataFrame):
    """
    Return the actual dollar volume in million USD
    
    Args:
        df (pd.DataFrame): daily dataframe of yfinance format.
    
    Returns:
        Dollar volume
    
    """
    return (df['adj close']*df['volume'])/1e6


def get_famafrench_factors(df: pd.DataFrame,
                           timeperiod:str = "M",
                           min_num_month:int = 10,
                           window_betas:int = 24):
    """
    Retrieve Fama-French factors and calculate rolling betas.

    Args:
        df (pd.DataFrame): DataFrame containing stock returns.
        timeperiod (str, optional): Time period for resampling ("M" for monthly, "Y" for yearly). Defaults to "M".
        min_num_month (int, optional): Minimum number of months required for calculation. Defaults to 10.
        window_betas (int, optional): Window size for calculating rolling betas. Defaults to 24.

    Returns:
        pd.DataFrame: DataFrame containing rolling betas for each stock.

    """
    
    if timeperiod == "M":
        factor_data = web.DataReader('F-F_Research_Data_Factors',
                                'famafrench',
                                start='2010')[0].drop('RF', axis=1)
    if timeperiod == "Y":
        factor_data = web.DataReader('F-F_Research_Data_Factors',
                                'famafrench',
                                start='2010')[1].drop('RF', axis=1)

    factor_data.index = factor_data.index.to_timestamp()

    factor_data = factor_data.resample(timeperiod).last().div(100)
    
    factor_data.index.name = df.index.names[0]
    
    factor_data = factor_data.join(df['return_1m']).sort_index()
    
    observations = factor_data.groupby(level=1).size()

    valid_stocks = observations[observations >= min_num_month]

    factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

    betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(window_betas, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

    return betas


class StockMeasures:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.measures = data.drop(data.columns, axis= 1)

    def calculate_measures(self, measure_names: str | list = 'all'):
        if measure_names == 'all' or 'garman_klass_vol' in measure_names:
            self.measures['garman_klass_vol'] = garman_klass_volatility(self.data)
        
        if measure_names == 'all' or 'rsi' in measure_names:
            self.measures['rsi'] = rsi(self.data, length= 20)

        if measure_names == 'all' or 'bollinger_bands' in measure_names:
            self.measures['bb_low'] =  bollinger_band(self.data, length= 20, index= 0)
            self.measures['bb_mid'] =  bollinger_band(self.data, length= 20, index= 1)
            self.measures['bb_high'] =  bollinger_band(self.data, length= 20, index= 2)
        
        if measure_names == 'all' or 'atr' in measure_names:
            self.measures['atr'] = norm_atr(self.data)
        
        if measure_names == 'all' or 'macd' in measure_names:
            self.measures['macd'] = norm_atr(self.data)

        if measure_names == 'all' or 'dollar_volume' in measure_names:
            self.measures['dollar_volume'] = dollar_volume(self.data)
 
 
 
    