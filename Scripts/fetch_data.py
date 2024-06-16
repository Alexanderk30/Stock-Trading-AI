import pandas as pd
import yfinance as yf

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f'{ticker}.csv')


fetch_data('AAPL', '2020-01-01', '2023-01-01')
