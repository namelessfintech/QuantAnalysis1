

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

# 2. Relative Strength Index RSI(14) Calculation and Chart
# Technical Indicator Calculation
aapl['rsi'] = ta.RSI(np.asarray(aapl['Close']), timeperiod=14)
# Technical Indicator Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Relative Strength Index RSI(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['rsi'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Bands Crossover Trading Signals
# Previous Periods Data (avoid back-testing bias)
aapl['rsi(-1)'] = aapl['rsi'].shift(1)
aapl['rsi(-2)'] = aapl['rsi'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['rsisig'] = 0
rsisig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['rsi(-2)'] < 30 and r[1]['rsi(-1)'] > 30:
        rsisig = 1
    elif r[1]['rsi(-2)'] < 70 and r[1]['rsi(-1)'] > 70:
        rsisig = -1
    else:
        rsisig = 0
    aapl.iloc[i, 9] = rsisig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Relative Strength Index RSI(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['rsi'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['rsisig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['rsistr'] = 1
rsistr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['rsisig'] == 1:
        rsistr = 1
    elif r[1]['rsisig'] == -1:
        rsistr = 0
    else:
        rsistr = aapl['rsistr'][i-1]
    aapl.iloc[i, 10] = rsistr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Relative Strength Index RSI(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['rsi'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['rsistr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price Crossover Strategy Without Trading Commissions
aapl['rsidrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['rsistr']
aapl.iloc[0, 11] = 0
# Price Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['rsistr(-1)'] = aapl['rsistr'].shift(1)
aapl['rsitc'] = aapl['rsisig']
rsitc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['rsisig'] == 1 or r[1]['rsisig'] == -1) and r[1]['rsistr'] != r[1]['rsistr(-1)']:
        rsitc = 0.01
    else:
        rsitc = 0.00
    aapl.iloc[i, 13] = rsitc
aapl['rsidrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['rsitc'])*aapl['rsistr']
aapl.iloc[0, 14] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 15] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['rsicrt'] = np.cumprod(aapl['rsidrt']+1)-1
aapl['rsicrtc'] = np.cumprod(aapl['rsidrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['rsicrt', 'rsicrtc', 'bhcrt'])
plt.title('Relative Strength Index RSI(14) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
rsiyrt = aapl.iloc[251, 16]
rsiyrtc = aapl.iloc[251, 17]
bhyrt = aapl.iloc[251, 18]
# Annualized Standard Deviation
rsistd = np.std(aapl['rsidrt'])*np.sqrt(252)
rsistdc = np.std(aapl['rsidrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
rsisr = rsiyrt/rsistd
rsisrc = rsiyrtc/rsistdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'RSI(14)', '2': 'RSI(14)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': rsiyrt, '2': rsiyrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': rsistd, '2': rsistdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': rsisr, '2': rsisrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)
