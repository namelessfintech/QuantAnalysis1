
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

# 2. Simple Moving Average SMA(5) and Relative Strength Index RSI(14) Calculation and Chart
# Technical Indicators Calculation
aapl['sma5'] = ta.SMA(np.asarray(aapl['Close']), 5)
aapl['rsi'] = ta.RSI(np.asarray(aapl['Close']), timeperiod=14)
# Technical Indicators Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Relative Strength Index RSI(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['rsi'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Price and Bands Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['Close(-1)'] = aapl['Close'].shift(1)
aapl['sma5(-1)'] = aapl['sma5'].shift(1)
aapl['rsi(-1)'] = aapl['rsi'].shift(1)
aapl['Close(-2)'] = aapl['Close'].shift(2)
aapl['sma5(-2)'] = aapl['sma5'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['rsismasig'] = 0
rsismasig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['Close(-2)'] < r[1]['sma5(-2)'] and r[1]['Close(-1)'] > r[1]['sma5(-1)'] and r[1]['rsi(-1)'] < 30:
        rsismasig = 1
    elif r[1]['Close(-2)'] > r[1]['sma5(-2)'] and r[1]['Close(-1)'] < r[1]['sma5(-1)'] and r[1]['rsi(-1)'] > 70:
        rsismasig = -1
    else:
        rsismasig = 0
    aapl.iloc[i, 13] = rsismasig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Relative Strength Index RSI(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['rsi'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['rsismasig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Price and Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['rsismastr'] = 1
rsismastr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['rsismasig'] == 1:
        rsismastr = 1
    elif r[1]['rsismasig'] == -1:
        rsismastr = 0
    else:
        rsismastr = aapl['rsismastr'][i-1]
    aapl.iloc[i, 14] = rsismastr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Relative Strength Index RSI(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['rsi'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['rsismastr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Price and Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price and Bands Crossover Strategy Without Trading Commissions
aapl['rsismadrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['rsismastr']
aapl.iloc[0, 15] = 0
# Price and Bands Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['rsismastr(-1)'] = aapl['rsismastr'].shift(1)
aapl['rsismatc'] = aapl['rsismasig']
rsismatc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['rsismasig'] == 1 or r[1]['rsismasig'] == -1) and r[1]['rsismastr'] != r[1]['rsismastr(-1)']:
        rsismatc = 0.01
    else:
        rsismatc = 0.00
    aapl.iloc[i, 17] = rsismatc
aapl['rsismadrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['rsismatc'])*aapl['rsismastr']
aapl.iloc[0, 18] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 19] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['rsismacrt'] = np.cumprod(aapl['rsismadrt']+1)-1
aapl['rsismacrtc'] = np.cumprod(aapl['rsismadrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['rsismacrt', 'rsismacrtc', 'bhcrt'])
plt.title('Simple Moving Average SMA(5) & Relative Strength Index RSI(14) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
rsismayrt = aapl.iloc[251, 20]
rsismayrtc = aapl.iloc[251, 21]
bhyrt = aapl.iloc[251, 22]
# Annualized Standard Deviation
rsismastd = np.std(aapl['rsismadrt'])*np.sqrt(252)
rsismastdc = np.std(aapl['rsismadrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
rsismasr = rsismayrt/rsismastd
rsismasrc = rsismayrtc/rsismastdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'SMA(5) &', '2': 'SMA(5) &', '3': 'B&H'},
        {'0': '', '1': 'RSI(14)', '2': 'RSI(14)TC', '3': ''},
        {'0': 'Annualized Return', '1': rsismayrt, '2': rsismayrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': rsismastd, '2': rsismastdc, '3': bhstd},
        {'0': 'Annualized: Sharpe Ratio (Rf=0%)', '1': rsismasr, '2': rsismasrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)