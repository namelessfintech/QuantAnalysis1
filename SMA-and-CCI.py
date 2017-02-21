########################################################################
##########################################################################

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

# 2. Simple Moving Average SMA(5) and Commodity Channel Index CCI(20,0.015) Calculation and Chart
# Technical Indicators Calculation
aapl['sma5'] = ta.SMA(np.asarray(aapl['Close']), 5)
aapl['cci'] = ta.CCI(np.asarray(aapl['High']), np.asarray(aapl['Low']), np.asarray(aapl['Close']), timeperiod=20)
# Technical Indicators Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Commodity Channel Index CCI(20,0.015)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['cci'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Price and Bands Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['Close(-1)'] = aapl['Close'].shift(1)
aapl['sma5(-1)'] = aapl['sma5'].shift(1)
aapl['cci(-1)'] = aapl['cci'].shift(1)
aapl['Close(-2)'] = aapl['Close'].shift(2)
aapl['sma5(-2)'] = aapl['sma5'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['ccismasig'] = 0
ccismasig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['Close(-2)'] < r[1]['sma5(-2)'] and r[1]['Close(-1)'] > r[1]['sma5(-1)'] and r[1]['cci(-1)'] < -100:
        ccismasig = 1
    elif r[1]['Close(-2)'] > r[1]['sma5(-2)'] and r[1]['Close(-1)'] < r[1]['sma5(-1)'] and r[1]['cci(-1)'] > 100:
        ccismasig = -1
    else:
        ccismasig = 0
    aapl.iloc[i, 13] = ccismasig
# Trading Stignals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Commodity Channel Index CCI(20,0.015)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['cci'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['ccismasig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Price and Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['ccismastr'] = 1
ccismastr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['ccismasig'] == 1:
        ccismastr = 1
    elif r[1]['ccismasig'] == -1:
        ccismastr = 0
    else:
        ccismastr = aapl['ccismastr'][i-1]
    aapl.iloc[i, 14] = ccismastr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Commodity Channel Index CCI(20,0.015)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['cci'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['ccismastr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Price and Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price and Bands Crossover Strategy Without Trading Commissions
aapl['ccismadrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['ccismastr']
aapl.iloc[0, 15] = 0
# Price and Bands Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['ccismastr(-1)'] = aapl['ccismastr'].shift(1)
aapl['ccismatc'] = aapl['ccismasig']
ccismatc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['ccismasig'] == 1 or r[1]['ccismasig'] == -1) and r[1]['ccismastr'] != r[1]['ccismastr(-1)']:
        ccismatc = 0.01
    else:
        ccismatc = 0.00
    aapl.iloc[i, 17] = ccismatc
aapl['ccismadrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['ccismatc'])*aapl['ccismastr']
aapl.iloc[0, 18] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 19] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['ccismacrt'] = np.cumprod(aapl['ccismadrt']+1)-1
aapl['ccismacrtc'] = np.cumprod(aapl['ccismadrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['ccismacrt', 'ccismacrtc', 'bhcrt'])
plt.title('Simple Moving Average SMA(5) & Commodity Channel Index CCI(20,0.015) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
ccismayrt = aapl.iloc[251, 20]
ccismayrtc = aapl.iloc[251, 21]
bhyrt = aapl.iloc[251, 22]
# Annualized Standard Deviation
ccismastd = np.std(aapl['ccismadrt'])*np.sqrt(252)
ccismastdc = np.std(aapl['ccismadrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
ccismasr = ccismayrt/ccismastd
ccismasrc = ccismayrtc/ccismastdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'SMA(5) &', '2': 'SMA(5) &', '3': 'B&H'},
        {'0': '', '1': 'CCI(20,0.015)', '2': 'CCI(20,0.015)TC', '3': ''},
        {'0': 'Annualized Return', '1': ccismayrt, '2': ccismayrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': ccismastd, '2': ccismastdc, '3': bhstd},
        {'0': 'Annualized: Sharpe Ratio (Rf=0%)', '1': ccismasr, '2': ccismasrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)