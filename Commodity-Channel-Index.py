######################################################
# Stock Technical Analysis with Python               #
# Commodity Channel Index CCI(20,0.015)              #
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

# 2. Commodity Channel Index CCI(20,0.015) Calculation and Chart

# Technical Indicator Calculation
aapl['cci'] = ta.CCI(np.asarray(aapl['High']), np.asarray(aapl['Low']), np.asarray(aapl['Close']), timeperiod=20)
# Technical Indicator Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Commodity Channel Index CCI(20,0.015)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['cci'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Bands Crossover Trading Signals
# Previous Periods Data (avoid back-testing bias)
aapl['cci(-1)'] = aapl['cci'].shift(1)
aapl['cci(-2)'] = aapl['cci'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['ccisig'] = 0
ccisig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['cci(-2)'] < -100 and r[1]['cci(-1)'] > -100:
        ccisig = 1
    elif r[1]['cci(-2)'] < 100 and r[1]['cci(-1)'] > 100:
        ccisig = -1
    else:
        ccisig = 0
    aapl.iloc[i, 9] = ccisig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Commodity Channel Index CCI(20,0.015)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['cci'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['ccisig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['ccistr'] = 1
ccistr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['ccisig'] == 1:
        ccistr = 1
    elif r[1]['ccisig'] == -1:
        ccistr = 0
    else:
        ccistr = aapl['ccistr'][i-1]
    aapl.iloc[i, 10] = ccistr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Commodity Channel Index CCI(20,0.015)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['cci'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['ccistr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price Crossover Strategy Without Trading Commissions
aapl['ccidrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['ccistr']
aapl.iloc[0, 11] = 0
# Price Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['ccistr(-1)'] = aapl['ccistr'].shift(1)
aapl['ccitc'] = aapl['ccisig']
ccitc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['ccisig'] == 1 or r[1]['ccisig'] == -1) and r[1]['ccistr'] != r[1]['ccistr(-1)']:
        ccitc = 0.01
    else:
        ccitc = 0.00
    aapl.iloc[i, 13] = ccitc
aapl['ccidrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['ccitc'])*aapl['ccistr']
aapl.iloc[0, 14] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 15] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['ccicrt'] = np.cumprod(aapl['ccidrt']+1)-1
aapl['ccicrtc'] = np.cumprod(aapl['ccidrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['ccicrt', 'ccicrtc', 'bhcrt'])
plt.title('Commodity Channel Index CCI(20,0.015) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metric
# Annualized Returns
cciyrt = aapl.iloc[251, 16]
cciyrtc = aapl.iloc[251, 17]
bhyrt = aapl.iloc[251, 18]
# Annualized Standard Deviation
ccistd = np.std(aapl['ccidrt'])*np.sqrt(252)
ccistdc = np.std(aapl['ccidrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
ccisr = cciyrt/ccistd
ccisrc = cciyrtc/ccistdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'CCI(20,0.015)', '2': 'CCI(20,0.015)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': cciyrt, '2': cciyrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': ccistd, '2': ccistdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': ccisr, '2': ccisrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)
