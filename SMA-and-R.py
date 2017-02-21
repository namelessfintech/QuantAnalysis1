

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

# 2. Simple Moving Average SMA(5) and Williams %R(14) Calculation and Chart
# Technical Indicators Calculation
aapl['sma5'] = ta.SMA(np.asarray(aapl['Close']), 5)
aapl['wpr'] = ta.WILLR(np.asarray(aapl['High']), np.asarray(aapl['Low']), np.asarray(aapl['Close']), timeperiod=14)
# Technical Indicators Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Williams %R(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['wpr'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Price and Bands Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['Close(-1)'] = aapl['Close'].shift(1)
aapl['sma5(-1)'] = aapl['sma5'].shift(1)
aapl['wpr(-1)'] = aapl['wpr'].shift(1)
aapl['Close(-2)'] = aapl['Close'].shift(2)
aapl['sma5(-2)'] = aapl['sma5'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['wprsmasig'] = 0
wprsmasig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['Close(-2)'] < r[1]['sma5(-2)'] and r[1]['Close(-1)'] > r[1]['sma5(-1)'] and r[1]['wpr(-1)'] < -80:
        wprsmasig = 1
    elif r[1]['Close(-2)'] > r[1]['sma5(-2)'] and r[1]['Close(-1)'] < r[1]['sma5(-1)'] and r[1]['wpr(-1)'] > -20:
        wprsmasig = -1
    else:
        wprsmasig = 0
    aapl.iloc[i, 13] = wprsmasig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Williams %R(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['wpr'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['wprsmasig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Price and Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['wprsmastr'] = 1
wprsmastr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['wprsmasig'] == 1:
        wprsmastr = 1
    elif r[1]['wprsmasig'] == -1:
        wprsmastr = 0
    else:
        wprsmastr = aapl['wprsmastr'][i-1]
    aapl.iloc[i, 14] = wprsmastr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Williams %R(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['wpr'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['wprsmastr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Price and Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price and Bands Crossover Strategy Without Trading Commissions
aapl['wprsmadrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['wprsmastr']
aapl.iloc[0, 15] = 0
# Price and Bands Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['wprsmastr(-1)'] = aapl['wprsmastr'].shift(1)
aapl['wprsmatc'] = aapl['wprsmasig']
wprsmatc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['wprsmasig'] == 1 or r[1]['wprsmasig'] == -1) and r[1]['wprsmastr'] != r[1]['wprsmastr(-1)']:
        wprsmatc = 0.01
    else:
        wprsmatc = 0.00
    aapl.iloc[i, 17] = wprsmatc
aapl['wprsmadrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['wprsmatc'])*aapl['wprsmastr']
aapl.iloc[0, 18] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 19] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['wprsmacrt'] = np.cumprod(aapl['wprsmadrt']+1)-1
aapl['wprsmacrtc'] = np.cumprod(aapl['wprsmadrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['wprsmacrt', 'wprsmacrtc', 'bhcrt'])
plt.title('Simple Moving Average SMA(5) & Williams %R(14) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
wprsmayrt = aapl.iloc[251, 20]
wprsmayrtc = aapl.iloc[251, 21]
bhyrt = aapl.iloc[251, 22]
# Annualized Standard Deviation
wprsmastd = np.std(aapl['wprsmadrt'])*np.sqrt(252)
wprsmastdc = np.std(aapl['wprsmadrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
wprsmasr = wprsmayrt/wprsmastd
wprsmasrc = wprsmayrtc/wprsmastdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'SMA(5) &', '2': 'SMA(5) &', '3': 'B&H'},
        {'0': '', '1': '%R(14)', '2': '%R(14)TC', '3': ''},
        {'0': 'Annualized Return', '1': wprsmayrt, '2': wprsmayrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': wprsmastd, '2': wprsmastdc, '3': bhstd},
        {'0': 'Annualized: Sharpe Ratio (Rf=0%)', '1': wprsmasr, '2': wprsmasrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)