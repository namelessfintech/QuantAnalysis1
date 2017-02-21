
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

# 2. Stochastic Oscillator Full STO(14,3,3) Calculation and Chart
# Technical Indicator Calculation
aapl['slowk'], aapl['slowd'] = ta.STOCH(np.asarray(aapl['High']), np.asarray(aapl['Low']), np.asarray(aapl['Close']),
                                     fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
# Technical Indicator Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Stochastic Oscillator Full STO(14,3,3)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['slowk'], color='r', linestyle='--')
aapl.plot(y=['slowd'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Bands Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['slowd(-1)'] = aapl['slowd'].shift(1)
aapl['slowd(-2)'] = aapl['slowd'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['stosig'] = 0
stosig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['slowd(-2)'] < 20 and r[1]['slowd(-1)'] > 20:
        stosig = 1
    elif r[1]['slowd(-2)'] < 80 and r[1]['slowd(-1)'] > 80:
        stosig = -1
    else:
        stosig = 0
    aapl.iloc[i, 10] = stosig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Stochastic Oscillator Full STO(14,3,3)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['slowk'], color='r', linestyle='--')
aapl.plot(y=['slowd'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['stosig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['stostr'] = 1
stostr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['stosig'] == 1:
        stostr = 1
    elif r[1]['stosig'] == -1:
        stostr = 0
    else:
        stostr = aapl['stostr'][i-1]
    aapl.iloc[i, 11] = stostr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Stochastic Oscillator Full STO(14,3,3)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['slowk'], color='r', linestyle='--')
aapl.plot(y=['slowd'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['stostr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price Crossover Strategy Without Trading Commissions
aapl['stodrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['stostr']
aapl.iloc[0, 12] = 0
# Price Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['stostr(-1)'] = aapl['stostr'].shift(1)
aapl['stotc'] = aapl['stosig']
stotc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['stosig'] == 1 or r[1]['stosig'] == -1) and r[1]['stostr'] != r[1]['stostr(-1)']:
        stotc = 0.01
    else:
        stotc = 0.00
    aapl.iloc[i, 14] = stotc
aapl['stodrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['stotc'])*aapl['stostr']
aapl.iloc[0, 15] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 16] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['stocrt'] = np.cumprod(aapl['stodrt']+1)-1
aapl['stocrtc'] = np.cumprod(aapl['stodrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['stocrt', 'stocrtc', 'bhcrt'])
plt.title('Stochastic Oscillator Full STO(14,3,3) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
stoyrt = aapl.iloc[251, 17]
stoyrtc = aapl.iloc[251, 18]
bhyrt = aapl.iloc[251, 19]
# Annualized Standard Deviation
stostd = np.std(aapl['stodrt'])*np.sqrt(252)
stostdc = np.std(aapl['stodrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
stosr = stoyrt/stostd
stosrc = stoyrtc/stostdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'STO(14,3,3)', '2': 'STO(14,3,3)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': stoyrt, '2': stoyrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': stostd, '2': stostdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': stosr, '2': stosrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)
