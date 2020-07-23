"""
Any generic calculations
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
import numpy as np
import pandas as pd


def returns(price_df, return_type, percent=False):
    """
    Calculate the returns of a price series.

    Arithmetic (simple) returns aggregate across assets.
    Geometric (log) returns aggregate across time.

    Parameters
    ----------
    price_df: pd.DataFrame
        Price time series to calculate returns of.
    return_type: str
        Either 'arithmetic' or 'geometric'.
        Type of returns to calculate.
    percent: bool
        Whether to give the results as a percentage rather than a decimal.
        Default False.

    Returns
    -------
    returns_df: pd.DataFrame
        Time series of returns per column of the input DataFrame.
    """
    if return_type == 'arithmetic':
        returns_df = arithmetic_returns(price_df)
    elif return_type == 'geometric':
        returns_df = geometric_returns(price_df)
    else:
        raise NotImplementedError(f"return_type must be 'arithmetic' or 'geometric'. "
                                  f"Got {return_type}")

    if percent:
        returns_df *= 100.

    return returns_df


def arithmetic_returns(price_df):
    """
    Calculate the arithmetic returns of a price series.

    This is calculated as:
    r(t) = p(t) / p(t-1) - 1

    Parameters
    ----------
    price_df: pd.DataFrame
        Price time series to calculate returns of.

    Returns
    -------
    pd.DataFrame
        Time series of returns per column of the input DataFrame.
    """
    return price_df / price_df.shift() - 1


def geometric_returns(price_df):
    """
    Calculate the geometric returns of a price series.

    This is calculated as:
    r(t) = ln(p(t) / p(t-1)) = ln(p(t)) - ln(p(t-1))

    Parameters
    ----------
    price_df: pd.DataFrame
        Price time series to calculate returns of.

    Returns
    -------
    pd.DataFrame
        Time series of returns per column of the input DataFrame.
    """
    return np.log(price_df) - np.log(price_df.shift())


# TODO TEST: check default values are sensible
def robust_volatility(df, span=35, min_periods=10, abs_floor=0.0000000001,
                      rolling_floor=True, floor_min_quant=0.05, floor_days=500,
                      floor_min_periods=100, backfill=False):
    """
    Robust exponential volatility calculation, assuming daily series of prices.

    We apply an absolute floor and a rolling floor based on recent history.

    Parameters
    ----------
    df: pd.DataFrame
        Time series to calculate volatility of.
    span: int
        Number of days in lookback for the N-day exponentially-weighted moving average.
    min_periods: int
        The minimum number of observations.
    abs_floor: float
        The size of absolute minimum.
    rolling_floor: bool
        Whether to apply a rolling quantile floor.
    floor_min_quant: float
        The quantile to use for the rolling floor, e.g. 0.05 corresponds to 5% vol.
    floor_days: int
        The number of days of lookback for calculating the rolling floor.
    floor_min_periods: int
        Minimum number of observations for the rolling floor. Until this is reached,
        the floor is zero.
    backfill: bool
        Whether to backfill the start of the timeseries.

    Returns
    -------
    vol_floored: pd.DataFrame
        Exponentially-weighted volatility with the given floors applied.
    """
    # Floor values at a minimum absolute value
    vol = df.ewm(span=span, min_periods=min_periods).std()
    vol_abs_floor = vol.clip(lower=abs_floor)

    if rolling_floor:
        vol_rolling_floor = (vol
                             .rolling(min_periods=floor_min_periods, window=floor_days)
                             .quantile(quantile=floor_min_quant))
        # Set this to zero for the first value then propagate forward, to ensure
        # we always have a value
        vol_rolling_floor.iloc[0, :] = 0.
        vol_rolling_floor = vol_rolling_floor.ffill()
        # Apply the rolling floor
        vol_floored = pd.concat([vol_abs_floor, vol_rolling_floor], axis=1)
        vol_floored = vol_floored.max(axis=1, skipna=False)
    else:
        vol_floored = vol_abs_floor

    if backfill:
        # Fill forwards first, as we only want to backfill NaNs at the start
        vol_floored = vol_floored.fillna(method="ffill").fillna(method="bfill")

    return vol_floored
