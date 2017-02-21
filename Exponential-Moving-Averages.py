

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

# 2. Exponential Moving Averages EMA(5 & 21) Calculation and Chart

# Technical Indicator Calculation
aapl['ema5'] = ta.EMA(np.asarray(aapl['Close']), 5)
aapl['ema21'] = ta.EMA(np.asarray(aapl['Close']), 21)
# Technical Indicator Chart
aapl.plot(y=['Close', 'ema5', 'ema21'])
plt.title('Apple Close Prices & Exponential Moving Averages EMA(5 & 21)')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Double Crossover Trading Signals
# Previous Periods Data (avoid back-testing bias)
aapl['ema5(-1)'] = aapl['ema5'].shift(1)
aapl['ema21(-1)'] = aapl['ema21'].shift(1)
aapl['ema5(-2)'] = aapl['ema5'].shift(2)
aapl['ema21(-2)'] = aapl['ema21'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['emasig'] = 0
emasig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['ema5(-2)'] < r[1]['ema21(-2)'] and r[1]['ema5(-1)'] > r[1]['ema21(-1)']:
        emasig = 1
    elif r[1]['ema5(-2)'] > r[1]['ema21(-2)'] and r[1]['ema5(-1)'] < r[1]['ema21(-1)']:
        emasig = -1
    else:
        emasig = 0
    aapl.iloc[i, 12] = emasig
# Trading Signals Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Exponential Moving Averages EMA(5 & 21)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'ema5', 'ema21'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['emasig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Double Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['emastr'] = 1
emastr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['emasig'] == 1:
        emastr = 1
    elif r[1]['emasig'] == -1:
        emastr = 0
    else:
        emastr = aapl['emastr'][i-1]
    aapl.iloc[i, 13] = emastr
# Trading Strategy Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Exponential Moving Averages EMA(5 & 21)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'ema5', 'ema21'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['emastr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Double Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Double Crossover Strategy Without Trading Commissions
aapl['emadrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['emastr']
aapl.iloc[0, 14] = 0
# Double Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['emastr(-1)'] = aapl['emastr'].shift(1)
aapl['ematc'] = aapl['emasig']
ematc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['emasig'] == 1 or r[1]['emasig'] == -1) and r[1]['emastr'] != r[1]['emastr(-1)']:
        ematc = 0.01
    else:
        ematc = 0.00
    aapl.iloc[i, 16] = ematc
aapl['emadrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['ematc'])*aapl['emastr']
aapl.iloc[0, 17] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 18] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['emacrt'] = np.cumprod(aapl['emadrt']+1)-1
aapl['emacrtc'] = np.cumprod(aapl['emadrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['emacrt', 'emacrtc', 'bhcrt'])
plt.title('Exponential Moving Averages EMA(5 & 21) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
emayrt = aapl.iloc[251, 19]
emayrtc = aapl.iloc[251, 20]
bhyrt = aapl.iloc[251, 21]
# Annualized Standard Deviation
emastd = np.std(aapl['emadrt'])*np.sqrt(252)
emastdc = np.std(aapl['emadrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
emasr = emayrt/emastd
emasrc = emayrtc/emastdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'EMA(5 & 21)', '2': 'EMA(5 & 21)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': emayrt, '2': emayrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': emastd, '2': emastdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': emasr, '2': emasrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)
