import pandas as pd


def aggregate_data_by_timestep(data: pd.DataFrame, timestep: str = "M"):
    """    
    The aggregation is done using the mean of the variable dollar 
    volume while using the last value of the timestep for the other indicators.

    Args:
        data (pd.DataFrame): dataframe of the daily measures.
        timestep (str, optional): Defaults to "M". 

    Returns:
        aggregated_dataframe: the resulting dataframe after the time period aggregation.
    """
    
    aggregated_dataframe = (
    pd.concat([
        data.unstack('ticker')['dollar_volume'].resample(timestep).mean().stack('ticker').to_frame('dollar_volume'),
        data.loc[:, ~data.columns.isin(["dollar_volume"])].unstack().resample(timestep).last().stack('ticker')
    ], 
    axis=1)).dropna()
    
    return aggregated_dataframe


def cut_low_volume(df: pd.DataFrame, window= 5, min_years= 1, threshold= 150):
    """
    We choose only the stocks that in 5 years have has a dollar volume exchange above 150 millions.
    This is done in order to get rid of rumorous data for our project purpose.

    Args:
        df (pd.DataFrame): the featured dataframe
        window (int, optional): period to run the rolling average . Defaults to 5 years.
        min_years (int, optional): minimum number of years. Defaults to 1.
        threshold (int, optional): minimum value of a volume stock exchange
                                    in millions of dollars. Defaults to 150.

    Returns:
        dataframe: dataframe of filtered stocks.
        
    """
    
    df['dollar_volume'] = (df.loc[:, 'dollar_volume'].unstack('ticker').rolling(window * 12, 
                                                                                min_periods= min_years * 12).mean().stack())

    df['dollar_vol_rank'] = (df.groupby('date')['dollar_volume'].rank(ascending=False))

    return  df[df['dollar_vol_rank'] < threshold].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)


def aggregate_returns(df: pd.DataFrame, outlier_cutoff: float= 0.005, lags: list= [1, 2, 3, 6, 9, 12]):
    """This function takes a DataFrame of aggregated financial features, calculates 
    returns for specified lag periods, handles outliers, and aggregates the results. 

    Args:
        df (pd.DataFrame): dataframe of aggregated features
        outlier_cutoff (float, optional): Cutoff value used to handle 
                                          outliers when calculating returns. Defaults to 0.005.
        lags (list, optional):  List of lag periods for which returns will be calculated.
                                Defaults to [1, 2, 3, 6, 9, 12].
    """
    
    def calculate_returns(df):
        for lag in lags:
            df[f'return_{lag}m'] = (df['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                        upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
        return df
    return df.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

