#!/usr/bin/env python3
"""preprocessor"""

import pandas as pd


def preprocess():
    """
    Preprocess function
    DataFrame cleaning
    Returns: train_set, valid_set, test_set
    """

    filename = './coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    df = pd.read_csv(filename)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.ffill()
    df = df[df['Timestamp'].dt.year >= 2017]
    df.index = df['Timestamp']
    df = df.resample('H').ffill()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df = df.drop(['Timestamp'], axis=1)
    df = df.drop(['Volume_(Currency)'], axis=1)
    df = df.dropna()
    df.reset_index(inplace=True, drop=True)

    # Save to csv
    # df.to_csv('cb_finBTC_preprocessed.csv', index=False)

    # Data set description
    print(df.describe().transpose())

    # Data set split train 70%, val 20%, test 10%
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]
    num_features = df.shape[1]

    print('Shape of train')
    print(train_df.shape)
    print('Shape of valid')
    print(val_df.shape)
    print('Shape of test')
    print(test_df.shape)
    print('Num of feats')
    print(num_features)

    # Normalization
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # for multi-step multivariate models
    num_features = df.shape[1]

    return train_df, val_df, test_df
