import backtrader as bt
import pandas as pd
import yfinance as yf

class MyStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        # Example strategy: Buy if price is above a certain threshold
        if self.dataclose[0] > 1.05 * self.dataclose[-1]:
            self.buy(size=10)

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True))
    cerebro.adddata(data)
    cerebro.addstrategy(MyStrategy)
    cerebro.run()
    cerebro.plot()
