

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

# 2. Rate of Change ROC(21) Calculation and Chart

# Technical Indicator Calculation
aapl['roc'] = ta.ROC(np.asarray(aapl['Close']), timeperiod=21)
# Technical Indicator Chart
plt.subplot(2, 1, 1)
plt.title('Apple Close Prices & Rate of Change ROC(21)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
aapl.plot(y=['roc'], color='g')
plt.legend(loc='upper left')
plt.show()

##########

# 3. Bands Crossover Trading Signals
# Previous Periods Data (avoid backtesting bias)
aapl['roc(-1)'] = aapl['roc'].shift(1)
aapl['roc(-2)'] = aapl['roc'].shift(2)
# Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
aapl['rocsig'] = 0
rocsig = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['roc(-2)'] < -10 and r[1]['roc(-1)'] > -10:
        rocsig = 1
    elif r[1]['roc(-2)'] < 10 and r[1]['roc(-1)'] > 10:
        rocsig = -1
    else:
        rocsig = 0
    aapl.iloc[i, 9] = rocsig
# Trading Signals Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Rate of Change ROC(21)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['roc'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['rocsig'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 4. Bands Crossover Trading Strategy
# Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available)
aapl['rocstr'] = 1
rocstr = 0
for i, r in enumerate(aapl.iterrows()):
    if r[1]['rocsig'] == 1:
        rocstr = 1
    elif r[1]['rocsig'] == -1:
        rocstr = 0
    else:
        rocstr = aapl['rocstr'][i-1]
    aapl.iloc[i, 10] = rocstr
# Trading Strategy Chart
plt.subplot(3, 1, 1)
plt.title('Apple Close Prices & Rate of Change ROC(21)')
plt.gca().axes.get_xaxis().set_visible(False)
aapl.plot(y=['Close'])
plt.legend(loc='upper left')
plt.subplot(3, 1, 2)
aapl.plot(y=['roc'], color='g')
plt.legend(loc='upper left')
plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(3, 1, 3)
aapl.plot(y=['rocstr'], marker='o', linestyle='')
plt.legend(loc='upper left')
plt.show()

##########

# 5. Bands Crossover Strategy Performance Comparison

# 5.1. Strategies Daily Returns
# Price Crossover Strategy Without Trading Commissions
aapl['rocdrt'] = ((aapl['Close']/aapl['Close'].shift(1))-1)*aapl['rocstr']
aapl.iloc[0, 11] = 0
# Price Crossover Strategy With Trading Commissions (1% Per Trade)
aapl['rocstr(-1)'] = aapl['rocstr'].shift(1)
aapl['roctc'] = aapl['rocsig']
roctc = 0
for i, r in enumerate(aapl.iterrows()):
    if (r[1]['rocsig'] == 1 or r[1]['rocsig'] == -1) and r[1]['rocstr'] != r[1]['rocstr(-1)']:
        roctc = 0.01
    else:
        roctc = 0.00
    aapl.iloc[i, 13] = roctc
aapl['rocdrtc'] = (((aapl['Close']/aapl['Close'].shift(1))-1)-aapl['roctc'])*aapl['rocstr']
aapl.iloc[0, 14] = 0
# Buy and Hold Strategy
aapl['bhdrt'] = (aapl['Close']/aapl['Close'].shift(1))-1
aapl.iloc[0, 15] = 0

# 5.2. Strategies Cumulative Returns
# Cumulative Returns Calculation
aapl['roccrt'] = np.cumprod(aapl['rocdrt']+1)-1
aapl['roccrtc'] = np.cumprod(aapl['rocdrtc']+1)-1
aapl['bhcrt'] = np.cumprod(aapl['bhdrt']+1)-1
# Cumulative Returns Chart
aapl.plot(y=['roccrt', 'roccrtc', 'bhcrt'])
plt.title('Rate of Change ROC(21) vs Buy & Hold')
plt.legend(loc='upper left')
plt.show()

# 5.3. Strategies Performance Metrics
# Annualized Returns
rocyrt = aapl.iloc[251, 16]
rocyrtc = aapl.iloc[251, 17]
bhyrt = aapl.iloc[251, 18]
# Annualized Standard Deviation
rocstd = np.std(aapl['rocdrt'])*np.sqrt(252)
rocstdc = np.std(aapl['rocdrtc'])*np.sqrt(252)
bhstd = np.std(aapl['bhdrt'])*np.sqrt(252)
# Annualized Sharpe Ratio
rocsr = rocyrt/rocstd
rocsrc = rocyrtc/rocstdc
bhsr = bhyrt/bhstd
# Summary Results Data Table
data = [{'0': '', '1': 'ROC(21)', '2': 'ROC(21)TC', '3': 'B&H'},
        {'0': 'Annualized Return', '1': rocyrt, '2': rocyrtc, '3': bhyrt},
        {'0': 'Annualized Standard Deviation', '1': rocstd, '2': rocstdc, '3': bhstd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': rocsr, '2': rocsrc, '3': bhsr}]
table = pd.DataFrame(data)
print(aapl)
print(table)
