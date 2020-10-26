#!/usr/bin/env python3
import pandas as pd


def from_file(filename, delimiter):
    """
    function from_file
    Args:
        filename: file to load from
        delimiter: column separator
    Returns: loaded pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
