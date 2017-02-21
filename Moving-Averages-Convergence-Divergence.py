
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

# 2. Moving Averages Convergence Divergence MACD(12,26,9) Calculation and Chart

# Technical Indicator Calculation
aapl['macd'], aapl['macdema'], aapl['macdhist'] = ta.MACD(np.asarray(aapl['Close']),
                                                      fastperiod=12, slowperiod=26, signalperiod=9)
# Technical Indicator Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Moving Averages Convergence Divergence MACD(12,26,9)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['macd', 'macdema'])
aapl.plot(y=['macdhist'], linestyle='--')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Signal Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['macd(-1)'] = aapl['macd'].shift(1)
aapl['macdema(-1)'] = aapl['macdema'].shift(1)
aapl['macd(-2)'] = aapl['macd'].shift(2)
aapl['macdema(-2)'] = aapl['macdema'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['macdsig'] = 0
macdsig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['macd(-2)'] < r[1]['macdema(-2)'] and r[1]['macd(-1)'] > r[1]['macdema(-1)']:
        macdsig = 1
    elif r[1]['macd(-2)'] > r[1]['macdema(-2)'] and r[1]['macd(-1)'] < r[1]['macdema(-1)']:
        macdsig = -1
    else:
        macdsig = 0
    aapl.iloc[i, 13] = macdsig
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Moving Averages Convergence Divergence MACD(12,26,9)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['macd', 'macdema'])
aapl.plot(y=['macdhist'], linestyle='--')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['macdsig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Signal Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['macdstr'] = 1
macdstr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['macdsig'] == 1:
        macdstr = 1
    elif r[1]['macdsig'] == -1:
        macdstr = 0
    else:
        macdstr = aapl['macdstr'][i-1]
    aapl.iloc[i, 14] = macdstr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Moving Averages Convergence Divergence MACD(12,26,9)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['macd', 'macdema'])
aapl.plot(y=['macdhist'], linestyle='--')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['macdstr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Signal Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Signal Crossover Strategy Without Trading Commissions
aapl['macddrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['macdstr']
aapl.iloc[0, 15] = 0
# Price Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['macdstr(-1)'] = aapl['macdstr'].shift(1)
aapl['macdtc'] = aapl['macdsig']
macdtc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['macdsig'] == 1 or r[1]['macdsig'] == -1) and r[1]['macdstr'] != r[1]['macdstr(-1)']:
        macdtc = 0.01
    else:
        macdtc = 0.00
    aapl.iloc[i, 17] = macdtc
aapl['macddrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['macdtc'])*aapl['macdstr']
aapl.iloc[0, 18] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 19] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['macdcrt'] = np.cumprod(aapl['macddrt']+1)-1
aapl['macdcrtc'] = np.cumprod(aapl['macddrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['macdcrt', 'macdcrtc', 'bhcrt'])
plt.title('Moving Averages Convergence Divergence MACD(12,26,9) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
macdyrt = aapl.iloc[251, 20]
macdyrtc = aapl.iloc[251, 21]
bhyrt = aapl.iloc[251, 22]
# Annualized Standard Deviation
macdstd = np.std(aapl['macddrt'])*np.sqrt(252)
macdstdc = np.std(aapl['macddrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
macdsr = macdyrt/macdstd
macdsrc = macdyrtc/macdstdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'MACD(12,26,9)', '2': 'MACD(12,26,9)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': macdyrt, '2': macdyrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': macdstd, '2': macdstdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': macdsr, '2': macdsrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)
