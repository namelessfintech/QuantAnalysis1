
######################################################

# 1. Packages and Data

# Packages Import
import numpy as np
import pandas as pd
import pandas.io.data as web
import datetime as dt
import matplotlib.pyplot as plt
import talib as ta
# Data Download
start = dt.datetime(2014, 10, 1)
end = dt.datetime(2015, 9, 30)
aapl = web.DataReader('AAPL', 'yahoo', start, end)

##########

# 2. Simple Moving Average SMA(5) Calculation and Chart
# Technical Indicator Calculation
aapl['sma5'] = ta.SMA(np.asarray(aapl['Close']), 5)
# Technical Indicator Chart
aapl.plot(y=['Close', 'sma5'])
plt.title('Apple Close Prices & Simple Moving Average SMA(5)')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Price Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['Close(-1)'] = aapl['Close'].shift(1)
aapl['sma5(-1)'] = aapl['sma5'].shift(1)
aapl['Close(-2)'] = aapl['Close'].shift(2)
aapl['sma5(-2)'] = aapl['sma5'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['sma5sig'] = 0
sma5sig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['Close(-2)'] < r[1]['sma5(-2)'] and r[1]['Close(-1)'] > r[1]['sma5(-1)']:
        sma5sig = 1
    elif r[1]['Close(-2)'] > r[1]['sma5(-2)'] and r[1]['Close(-1)'] < r[1]['sma5(-1)']:
        sma5sig = -1
    else:
        sma5sig = 0
    aapl.iloc[i, 11] = sma5sig
# Trading Signals Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Simple Moving Average SMA(5)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['sma5sig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Price Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['sma5str'] = 1
sma5str = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['sma5sig'] == 1:
        sma5str = 1
    elif r[1]['sma5sig'] == -1:
        sma5str = 0
    else:
        sma5str = aapl['sma5str'][i-1]
    aapl.iloc[i, 12] = sma5str
# Trading Strategy Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Simple Moving Average SMA(5)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['sma5str'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Price Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price Crossover Strategy Without Trading Commissions
aapl['sma5drt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['sma5str']
aapl.iloc[0, 13] = 0
# Price Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['sma5str(-1)'] = aapl['sma5str'].shift(1)
aapl['sma5tc'] = aapl['sma5sig']
sma5tc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['sma5sig'] == 1 or r[1]['sma5sig'] == -1) and r[1]['sma5str'] != r[1]['sma5str(-1)']:
        sma5tc = 0.01
    else:
        sma5tc = 0.00
    aapl.iloc[i, 15] = sma5tc
aapl['sma5drtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['sma5tc'])*aapl['sma5str']
aapl.iloc[0, 16] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 17] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['sma5crt'] = np.cumprod(aapl['sma5drt']+1)-1
aapl['sma5crtc'] = np.cumprod(aapl['sma5drtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['sma5crt', 'sma5crtc', 'bhcrt'])
plt.title('Simple Moving Average SMA(5) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
sma5yrt = aapl.iloc[251, 18]
sma5yrtc = aapl.iloc[251, 19]
bhyrt = aapl.iloc[251, 20]
# Annualized Standard Deviation
sma5std = np.std(aapl['sma5drt'])*np.sqrt(252)
sma5stdc = np.std(aapl['sma5drtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
sma5sr = sma5yrt/sma5std
sma5src = sma5yrtc/sma5stdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'SMA(5)', '2': 'SMA(5)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': sma5yrt, '2': sma5yrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': sma5std, '2': sma5stdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': sma5sr, '2': sma5src, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)



