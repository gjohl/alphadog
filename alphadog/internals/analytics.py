"""
internals: any generic calculations
-- volatility
-- beta
-- correlation
-- covariance
-- ewma
-- kalman filter
-- state space models
-- ML libraries...
-- Sharpe ratio
-- drawdowns
-- autocorrelations
-- Granger causality
"""
import pandas as pd


# TODO: add a robust vol calc that floors the vol at a rolling min value


def cross_sectional_mean(df, name='combined'):
    """
    Aggregate multiple columns by taking the cross-sectional mean.

    Parameters
    ----------
    df: pd.DataFrame

    name

    Returns
    -------

    """
    return df.mean(axis=1).to_frame(name)
