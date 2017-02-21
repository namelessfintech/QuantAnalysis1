# Packages Import
import numpy as np
import pandas as pd
from pandas_datareader import data
import datetime as dt
import matplotlib.pyplot as plt
import talib as ta
import csv
# Data Download
start = dt.datetime(2017, 1, 1)
end = dt.datetime(2017, 2, 19)
glw = data.DataReader('GLW', 'yahoo', start, end)

glw.to_csv('/Users/MichaelBallard/desktop/hls.csv')
print(glw)