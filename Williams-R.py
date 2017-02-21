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

# 2. Williams %R(14) Calculation and Chart
# Technical Indicator Calculation
aapl['wpr'] = ta.WILLR(np.asarray(aapl['High']), np.asarray(aapl['Low']), np.asarray(aapl['Close']), timeperiod=14)
# Technical Indicator Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Williams %R(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['wpr'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Bands Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['wpr(-1)'] = aapl['wpr'].shift(1)
aapl['wpr(-2)'] = aapl['wpr'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['wprsig'] = 0
wprsig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['wpr(-2)'] < -80 and r[1]['wpr(-1)'] > -80:
        wprsig = 1
    elif r[1]['wpr(-2)'] < -20 and r[1]['wpr(-1)'] > -20:
        wprsig = -1
    else:
        wprsig = 0
    aapl.iloc[i, 9] = wprsig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Williams %R(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['wpr'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['wprsig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()


##########

# 4. Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['wprstr'] = 1
wprstr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['wprsig'] == 1:
        wprstr = 1
    elif r[1]['wprsig'] == -1:
        wprstr = 0
    else:
        wprstr = aapl['wprstr'][i-1]
    aapl.iloc[i, 10] = wprstr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Williams %R(14)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['wpr'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['wprstr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price Crossover Strategy Without Trading Commissions
aapl['wprdrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['wprstr']
aapl.iloc[0, 11] = 0
# Price Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['wprstr(-1)'] = aapl['wprstr'].shift(1)
aapl['wprtc'] = aapl['wprsig']
wprtc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['wprsig'] == 1 or r[1]['wprsig'] == -1) and r[1]['wprstr'] != r[1]['wprstr(-1)']:
        wprtc = 0.01
    else:
        wprtc = 0.00
    aapl.iloc[i, 13] = wprtc
aapl['wprdrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['wprtc'])*aapl['wprstr']
aapl.iloc[0, 14] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 15] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['wprcrt'] = np.cumprod(aapl['wprdrt']+1)-1
aapl['wprcrtc'] = np.cumprod(aapl['wprdrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['wprcrt', 'wprcrtc', 'bhcrt'])
plt.title('Williams %R(14) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
wpryrt = aapl.iloc[251, 16]
wpryrtc = aapl.iloc[251, 17]
bhyrt = aapl.iloc[251, 18]
# Annualized Standard Deviation
wprstd = np.std(aapl['wprdrt'])*np.sqrt(252)
wprstdc = np.std(aapl['wprdrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
wprsr = wpryrt/wprstd
wprsrc = wpryrtc/wprstdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': '%R(14)', '2': '%R(14)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': wpryrt, '2': wpryrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': wprstd, '2': wprstdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': wprsr, '2': wprsrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)
