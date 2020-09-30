#!/usr/bin/env python3

import pandas as pd

def preprocess():
    filename = './coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    df = pd.read_csv(filename)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.ffill()
    df = df[df['Timestamp'].dt.year >= 2015]
    df.index = df['Timestamp']
    df = df.resample('H').ffill()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df = df.drop(['Timestamp'], axis=1)
    df = df.dropna()
    df.reset_index(inplace=True, drop=True)

    df.to_csv('cb_finBTC_preprocessed.csv', index=False)
    return df