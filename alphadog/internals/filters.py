"""
Generic timeseries filters.
"""

import numpy as np
import pandas as pd


def cusum_filter(df, threshold):
    """
    Event-based sampling filter to detects shifts away from the mean in
    a locally stationary process.

    Parameters
    ----------
    df: pd.DataFrame
        Time series to detect changes in.
    threshold: float
        Deviation from the mean required to trigger a positive event.

    Returns
    -------
    pd.DataFrame
        Binary time series of positive events.
        1 indicates a trigger event, 0 otherwise.

    References
    ----------
    Lopez de Prado, Advances in Financial Machine Learning (2018), pp 38-40
    """
    pass


def kalman_filter():
    # TODO
    pass