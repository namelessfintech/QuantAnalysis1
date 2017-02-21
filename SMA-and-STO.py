
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

# 2. Simple Moving Average SMA(5) and Stochastic Oscillator Full STO(14,3,3) Calculation and Chart
# Technical Indicators Calculation
aapl['sma5'] = ta.SMA(np.asarray(aapl['Close']), 5)
aapl['slowk'], aapl['slowd'] = ta.STOCH(np.asarray(aapl['High']), np.asarray(aapl['Low']), np.asarray(aapl['Close']),
                                     fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
# Technical Indicators Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Stochastic Oscillator Full STO(14,3,3)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['slowk'], color='r', linestyle='--')
aapl.plot(y=['slowd'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Price and Bands Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['Close(-1)'] = aapl['Close'].shift(1)
aapl['sma5(-1)'] = aapl['sma5'].shift(1)
aapl['slowd(-1)'] = aapl['slowd'].shift(1)
aapl['Close(-2)'] = aapl['Close'].shift(2)
aapl['sma5(-2)'] = aapl['sma5'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['stosmasig'] = 0
stosmasig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['Close(-2)'] < r[1]['sma5(-2)'] and r[1]['Close(-1)'] > r[1]['sma5(-1)'] and r[1]['slowd(-1)'] < 20:
        stosmasig = 1
    elif r[1]['Close(-2)'] > r[1]['sma5(-2)'] and r[1]['Close(-1)'] < r[1]['sma5(-1)'] and r[1]['slowd(-1)'] > 80:
        stosmasig = -1
    else:
        stosmasig = 0
    aapl.iloc[i, 14] = stosmasig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Stochastic Oscillator Full STO(14,3,3)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['slowk'], color='r', linestyle='--')
aapl.plot(y=['slowd'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['stosmasig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()


##########

# 4. Price and Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['stosmastr'] = 1
stosmastr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['stosmasig'] == 1:
        stosmastr = 1
    elif r[1]['stosmasig'] == -1:
        stosmastr = 0
    else:
        stosmastr = aapl['stosmastr'][i-1]
    aapl.iloc[i, 15] = stosmastr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices, Simple Moving Average SMA(5) & Stochastic Oscillator Full STO(14,3,3)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'sma5'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['slowk'], color='r', linestyle='--')
aapl.plot(y=['slowd'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['stosmastr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Price and Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price and Bands Crossover Strategy Without Trading Commissions
aapl['stosmadrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['stosmastr']
aapl.iloc[0, 16] = 0
# Price and Bands Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['stosmastr(-1)'] = aapl['stosmastr'].shift(1)
aapl['stosmatc'] = aapl['stosmasig']
stosmatc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['stosmasig'] == 1 or r[1]['stosmasig'] == -1) and r[1]['stosmastr'] != r[1]['stosmastr(-1)']:
        stosmatc = 0.01
    else:
        stosmatc = 0.00
    aapl.iloc[i, 18] = stosmatc
aapl['stosmadrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['stosmatc'])*aapl['stosmastr']
aapl.iloc[0, 19] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 20] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['stosmacrt'] = np.cumprod(aapl['stosmadrt']+1)-1
aapl['stosmacrtc'] = np.cumprod(aapl['stosmadrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['stosmacrt', 'stosmacrtc', 'bhcrt'])
plt.title('Simple Moving Average SMA(5) & Stochastic Oscillator Full STO(14,3,3) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
stosmayrt = aapl.iloc[251, 21]
stosmayrtc = aapl.iloc[251, 22]
bhyrt = aapl.iloc[251, 23]
# Annualized Standard Deviation
stosmastd = np.std(aapl['stosmadrt'])*np.sqrt(252)
stosmastdc = np.std(aapl['stosmadrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
stosmasr = stosmayrt/stosmastd
stosmasrc = stosmayrtc/stosmastdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'SMA(5) &', '2': 'SMA(5) &', '3': 'B&H'},
        {'0': '', '1': 'STO(14,3,3)', '2': 'STO(14,3,3)TC', '3': ''},
        {'0': 'Annualized Return', '1': stosmayrt, '2': stosmayrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': stosmastd, '2': stosmastdc, '3': bhstd},
        {'0': 'Annualized: Sharpe Ratio (Rf=0%)', '1': stosmasr, '2': stosmasrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)