
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

# 2. Simple Moving Average SMA(5) and Rate of Change ROC(21) Calculation and Chart
# Technical Indicators Calculation
aapl['sma5'] = ta.SMA(np.asarray(aapl['Close']), 5)
aapl['roc'] = ta.ROC(np.asarray(aapl['Close']), timeperiod=21)
# Technical Indicators Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Rate of Change ROC(21)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['roc'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Price and Bands Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['Close(-1)'] = aapl['Close'].shift(1)
aapl['sma5(-1)'] = aapl['sma5'].shift(1)
aapl['roc(-1)'] = aapl['roc'].shift(1)
aapl['Close(-2)'] = aapl['Close'].shift(2)
aapl['sma5(-2)'] = aapl['sma5'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['rocsmasig'] = 0
rocsmasig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['Close(-2)'] < r[1]['sma5(-2)'] and r[1]['Close(-1)'] > r[1]['sma5(-1)'] and r[1]['roc(-1)'] < -10:
        rocsmasig = 1
    elif r[1]['Close(-2)'] > r[1]['sma5(-2)'] and r[1]['Close(-1)'] < r[1]['sma5(-1)'] and r[1]['roc(-1)'] > 10:
        rocsmasig = -1
    else:
        rocsmasig = 0
    aapl.iloc[i, 13] = rocsmasig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Rate of Change ROC(21)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['roc'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['rocsmasig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()


##########

# 4. Price and Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['rocsmastr'] = 1
rocsmastr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['rocsmasig'] == 1:
        rocsmastr = 1
    elif r[1]['rocsmasig'] == -1:
        rocsmastr = 0
    else:
        rocsmastr = aapl['rocsmastr'][i-1]
    aapl.iloc[i, 14] = rocsmastr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Rate of Change ROC(21)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['roc'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['rocsmastr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Price and Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price and Bands Crossover Strategy Without Trading Commissions
aapl['rocsmadrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['rocsmastr']
aapl.iloc[0, 15] = 0
# Price and Bands Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['rocsmastr(-1)'] = aapl['rocsmastr'].shift(1)
aapl['rocsmatc'] = aapl['rocsmasig']
rocsmatc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['rocsmasig'] == 1 or r[1]['rocsmasig'] == -1) and r[1]['rocsmastr'] != r[1]['rocsmastr(-1)']:
        rocsmatc = 0.01
    else:
        rocsmatc = 0.00
    aapl.iloc[i, 17] = rocsmatc
aapl['rocsmadrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['rocsmatc'])*aapl['rocsmastr']
aapl.iloc[0, 18] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 19] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['rocsmacrt'] = np.cumprod(aapl['rocsmadrt']+1)-1
aapl['rocsmacrtc'] = np.cumprod(aapl['rocsmadrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['rocsmacrt', 'rocsmacrtc', 'bhcrt'])
plt.title('Simple Moving Average SMA(5) & Rate of Change ROC(21) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
rocsmayrt = aapl.iloc[251, 20]
rocsmayrtc = aapl.iloc[251, 21]
bhyrt = aapl.iloc[251, 22]
# Annualized Standard Deviation
rocsmastd = np.std(aapl['rocsmadrt'])*np.sqrt(252)
rocsmastdc = np.std(aapl['rocsmadrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
rocsmasr = rocsmayrt/rocsmastd
rocsmasrc = rocsmayrtc/rocsmastdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'SMA(5) &', '2': 'SMA(5) &', '3': 'B&H'},
        {'0': '', '1': 'ROC(21)', '2': 'ROC(21)TC', '3': ''},
        {'0': 'Annualized Return', '1': rocsmayrt, '2': rocsmayrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': rocsmastd, '2': rocsmastdc, '3': bhstd},
        {'0': 'Annualized: Sharpe Ratio (Rf=0%)', '1': rocsmasr, '2': rocsmasrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)