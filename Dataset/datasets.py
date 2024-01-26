import pandas as pd
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import financedatabase as fd
import logging

# Suppress warning messages from yfinance
logging.getLogger("yfinance").setLevel(logging.ERROR)

def download_stock_data(market_country: str = 'Italy', n_years: int = 5,
                        end_date: str = str(dt.datetime.today().date()),
                        start_date = None)-> pd.DataFrame:
    """
        params:
            market_country: the country name of the stocks to download
            n_years: the number of years we want the stocks
            end_date: the last date of the stock prices (default today)

        output:
            the dataframe containing daily stocks
    """
    
    if start_date == None:
        start_date = pd.to_datetime(end_date) - pd.DateOffset(365* n_years)
    else:
        start_date = pd.to_datetime(start_date)


    if market_country in ('IT', 'Italy'):
        
        equities = fd.Equities()
        equities = equities.search( currency='EUR', country='Italy', exchange='MIL')

        symbols_list = equities.index.unique().tolist() + ["ENI.MI"]
        
    elif market_country in ('US', 'United States'):
        
        
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

        symbols_list = sp500['Symbol'].unique().tolist()

    # dataframe creation 
    df = yf.download(tickers=symbols_list,
                start=start_date,
                end=end_date).stack()

    df.index.names = ['date', 'ticker']

    df.columns = df.columns.str.lower()

    return df