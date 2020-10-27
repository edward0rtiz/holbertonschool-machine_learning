#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = from_file("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", ',')

df = df.loc[df["Timestamp"] >= 1483228800]
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit='s')
df = df.set_index("Date")

df = df.drop(["Weighted_Price"], axis=1)
df["Open"].fillna(method='ffill', inplace=True)
df["High"].fillna(method='ffill', inplace=True)
df["Low"].fillna(method='ffill', inplace=True)
df["Close"].fillna(method='ffill', inplace=True)
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
df2 = pd.DataFrame()
df2["High"] = df.High.resample('D').max()
df2["Low"] = df.Low.resample('D').min()
df2["Volume_(BTC)"] = df["Volume_(BTC)"].resample('D').sum()
df2["Volume_(Currency)"] = df["Volume_(Currency)"].resample('D').sum()
df2["Open"] = df.Open.resample('D')
df2["close"] = df.Close.resample('D')
df2.plot()
plt.show()