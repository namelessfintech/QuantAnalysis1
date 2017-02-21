######################################################
# Stock Technical Analysis with Python               #
# Bollinger Bands BB(20,2)                           #
# (c) Diego Fernandez Garcia 2016                    #
# www.exfinsis.com                                   #
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

# 2. Bollinger Bands BB(20,2) Calculation and Chart

# Technical Indicator Calculation
aapl['upper'], aapl['middle'], aapl['lower'] = ta.BBANDS(np.asarray(aapl['Close']),
                                                     timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
# Technical Indicator Chart
aapl.plot(y=['Close', 'upper', 'middle', 'lower'])
plt.title('Apple Close Prices & Bollinger Bands BB(20,2)')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Bands Crossover Trading Signals
# Previous Periods Data (avoid back-testing bias)
aapl['Close(-1)'] = aapl['Close'].shift(1)
aapl['lower(-1)'] = aapl['lower'].shift(1)
aapl['upper(-1)'] = aapl['upper'].shift(1)
aapl['Close(-2)'] = aapl['Close'].shift(2)
aapl['lower(-2)'] = aapl['lower'].shift(2)
aapl['upper(-2)'] = aapl['upper'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['bbsig'] = 0
bbsig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['Close(-2)'] < r[1]['lower(-2)'] and r[1]['Close(-1)'] > r[1]['lower(-1)']:
        bbsig = 1
    elif r[1]['Close(-2)'] < r[1]['upper(-2)'] and r[1]['Close(-1)'] > r[1]['upper(-1)']:
        bbsig = -1
    else:
        bbsig = 0
    aapl.iloc[i, 15] = bbsig
# Trading Signals Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Bollinger Bands BB(20,2)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'upper', 'middle', 'lower'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['bbsig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['bbstr'] = 1
bbstr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['bbsig'] == 1:
        bbstr = 1
    elif r[1]['bbsig'] == -1:
        bbstr = 0
    else:
        bbstr = aapl['bbstr'][i-1]
    aapl.iloc[i, 16] = bbstr
# Trading Strategy Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Bollinger Bands BB(20,2)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close', 'upper', 'middle', 'lower'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['bbstr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Bands Crossover Strategy Without Trading Commissions
aapl['bbdrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['bbstr']
aapl.iloc[0, 17] = 0
# Bands Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['bbstr(-1)'] = aapl['bbstr'].shift(1)
aapl['bbtc'] = aapl['bbsig']
bbtc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['bbsig'] == 1 or r[1]['bbsig'] == -1) and r[1]['bbstr'] != r[1]['bbstr(-1)']:
        bbtc = 0.01
    else:
        bbtc = 0.00
    aapl.iloc[i, 19] = bbtc
aapl['bbdrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['bbtc'])*aapl['bbstr']
aapl.iloc[0, 20] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 21] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['bbcrt'] = np.cumprod(aapl['bbdrt']+1)-1
aapl['bbcrtc'] = np.cumprod(aapl['bbdrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['bbcrt', 'bbcrtc', 'bhcrt'])
plt.title('Bollinger Bands BB(20,2) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
bbyrt = aapl.iloc[251, 22]
bbyrtc = aapl.iloc[251, 23]
bhyrt = aapl.iloc[251, 24]
# Annualized Standard Deviation
bbstd = np.std(aapl['bbdrt'])*np.sqrt(252)
bbstdc = np.std(aapl['bbdrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
bbsr = bbyrt/bbstd
bbsrc = bbyrtc/bbstdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'BB(20,2)', '2': 'BB(20,2)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': bbyrt, '2': bbyrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': bbstd, '2': bbstdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': bbsr, '2': bbsrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)
