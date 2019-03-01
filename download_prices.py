import os
from pandas_datareader import data as pdr
import pandas as pd

import yaml

import fix_yahoo_finance as yf
yf.pdr_override()


def download_stock(ticker='VTSAX', start_date='2000-01-01', end_date='2019-01-01'):
    """Download the historical prices for this ticker
    """
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)

    return data

def get_all_stocks(ticker_list=['VTSAX', 'VIIIX'], start_date='2000-01-01', end_date='2019-01-01'):
    """Download all stocks and add to panda dataframe

    """
    data_frame = pd.DataFrame()

    for ticker in ticker_list:
        print("Downloading {}".format(ticker))
        data = download_stock(ticker, start_date, end_date)
        if data.empty:
            print("No data for {}".format(ticker))
            continue

        adj_close = data['Adj Close'].rename(ticker)
        data_frame = pd.concat([data_frame, adj_close], axis=1)

    data_frame.to_hdf("stock_prices.hdf5", "df")
    # data_frame.to_csv("stock_prices.csv")

    return 0

if __name__ == "__main__":
    # get list of stocks/funds
    with open('./stock_list.yml') as stocks:
        stock_list = yaml.load(stocks)

    get_all_stocks(stock_list)



