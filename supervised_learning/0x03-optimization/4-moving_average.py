#!/usr/bin/env python3
"""Script to calculate the weighted
    moving average of a data set
"""


def moving_average(data, beta):
    """
    Function to calulate Exponential Moving Average
    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average

    Returns: list containing the moving averages of data

    """
    Vt = 0
    EMA = []
    for i in range(len(data)):
        Vt = (beta * Vt) + ((1 - beta) * data[i])
        bias_correction = 1 - beta ** (i + 1)
        new_Vt = Vt / bias_correction
        EMA.append(new_Vt)
    return EMA
