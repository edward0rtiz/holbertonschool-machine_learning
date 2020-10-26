#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])
df["Open"].fillna(method='ffill', inplace=True)
df["High"].fillna(method='ffill', inplace=True)
df["Low"].fillna(method='ffill', inplace=True)
df["Close"].fillna(method='ffill', inplace=True)
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
print(df.head())
print(df.tail())