#!/usr/bin/env python3
import pandas as pd


def from_numpy(array):
    """
    function to create a dataframe
    Args:
        array: np.ndarray from which you should create the pd.DataFrame
    Returns: newly created pd.DataFrame
    """
    c_list = list('ABCDEFGH')
    reshape = c_list[:array.shape[1]]
    return pd.DataFrame(array, columns=reshape)