import pandas as pd
import datetime as dt

def list_monthly_stocks(df:pd.DataFrame):
    
    """_summary_

    Args:
        df (pd.DataFrame): dataframe containing all the stocks selected for each month.
        
    Returns:
        monthly_stocks (dict): the selected stocks for the following month.
    
    
    """
    # We use as a index only the dates
    
    filtered_df = df.reset_index(level=1)

    # We move each index one day in the future. So from the last day of the month
    # to the first day of the following month
    filtered_df.index = filtered_df.index + pd.DateOffset(1)
    
    # we reset the index as date and ticker
    filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

    dates = filtered_df.index.get_level_values('date').unique().tolist()

    choosen_stocks = {}

    for d in dates:
        
        choosen_stocks[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
        
    return choosen_stocks 