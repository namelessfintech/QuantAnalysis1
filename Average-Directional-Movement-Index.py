######################################################
# Stock Technical Analysis with Python               #
# Average Directional Movement Index ADX(14)         #
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

# 2. Average Directional Movement Index ADX(14) Calculation and Chart

# Technical Indicator Calculation
aapl['adx'] = ta.ADX(np.asarray(aapl['High']), np.asarray(aapl['Low']), np.asarray(aapl['Close']), timeperiod=14)
aapl['+di'] = ta.PLUS_DI(np.asarray(aapl['High']), np.asarray(aapl['Low']), np.asarray(aapl['Close']), timeperiod=14)
aapl['-di'] = ta.MINUS_DI(np.asarray(aapl['High']), np.asarray(aapl['Low']), np.asarray(aapl['Close']), timeperiod=14)
# Technical Indicator Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Average Directional Movement Index ADX(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['adx', '+di', '-di'])
plt.legend(loc='upper left')
plt.show()

##########

# 3. Bands and Double Crossover Trading Signals
# Previous Periods Data (avoid back-testing bias)
aapl['adx(-1)'] = aapl['adx'].shift(1)
aapl['+di(-1)'] = aapl['+di'].shift(1)
aapl['-di(-1)'] = aapl['-di'].shift(1)
aapl['+di(-2)'] = aapl['+di'].shift(2)
aapl['-di(-2)'] = aapl['-di'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['adxsig'] = 0
adxsig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['+di(-2)'] < r[1]['-di(-2)'] and r[1]['+di(-1)'] > r[1]['-di(-1)'] and r[1]['adx(-1)'] > 20:
        adxsig = 1
    elif r[1]['+di(-2)'] > r[1]['-di(-2)'] and r[1]['+di(-1)'] < r[1]['-di(-1)'] and r[1]['adx(-1)'] > 20:
        adxsig = -1
    else:
        adxsig = 0
    aapl.iloc[i, 14] = adxsig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Average Directional Movement Index ADX(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['adx', '+di', '-di'])
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['adxsig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Bands and Double Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['adxstr'] = 1
adxstr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['adxsig'] == 1:
        adxstr = 1
    elif r[1]['adxsig'] == -1:
        adxstr = 0
    else:
        adxstr = aapl['adxstr'][i-1]
    aapl.iloc[i, 15] = adxstr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Average Directional Movement Index ADX(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['adx', '+di', '-di'])
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['adxstr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Bands and Double Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Bands and Double Crossover Strategy Without Trading Commissions
aapl['adxdrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['adxstr']
aapl.iloc[0, 16] = 0
# Bands and Double Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['adxstr(-1)'] = aapl['adxstr'].shift(1)
aapl['adxtc'] = aapl['adxsig']
adxtc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['adxsig'] == 1 or r[1]['adxsig'] == -1) and r[1]['adxstr'] != r[1]['adxstr(-1)']:
        adxtc = 0.01
    else:
        adxtc = 0.00
    aapl.iloc[i, 18] = adxtc
aapl['adxdrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['adxtc'])*aapl['adxstr']
aapl.iloc[0, 19] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 20] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['adxcrt'] = np.cumprod(aapl['adxdrt']+1)-1
aapl['adxcrtc'] = np.cumprod(aapl['adxdrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['adxcrt', 'adxcrtc', 'bhcrt'])
plt.title('Average Directional Movement Index ADX(14) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
adxyrt = aapl.iloc[251, 21]
adxyrtc = aapl.iloc[251, 22]
bhyrt = aapl.iloc[251, 23]
# Annualized Standard Deviation
adxstd = np.std(aapl['adxdrt'])*np.sqrt(252)
adxstdc = np.std(aapl['adxdrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
adxsr = adxyrt/adxstd
adxsrc = adxyrtc/adxstdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'ADX(14)', '2': 'ADX(14)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': adxyrt, '2': adxyrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': adxstd, '2': adxstdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': adxsr, '2': adxsrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)
