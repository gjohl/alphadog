"""
Functionality to handle FX conversions.
"""
import pandas as pd

from alphadog.data.data_quality import (
    check_nonempty_dataframe, check_scalar_is_above_min_threshold
)
from alphadog.data.retrieval import PriceData
from alphadog.framework.constants import MAX_FFILL


def get_fx(from_ccy, to_ccy):
    """
    Get the FX rate to convert from one currency to another.

    Also handles conversions of pounds (GBP) and pence (GBX).

    Parameters
    ----------
    from_ccy: str
        Currency ISO code to convert from.
    to_ccy: str
        Currency ISO code to convert from.

    Returns
    -------
    pd.DataFrame
        Time series of FX rates.
        Returns a float if this is constant (e.g. 100 GBX = 1 GBP)
    """
    multiplier = None

    # Handle no conversion
    if from_ccy == to_ccy:
        return 1.

    # Handle GBX
    if from_ccy == 'GBX':
        if to_ccy == 'GBP':
            return 0.01
        else:
            from_ccy = 'GBP'
            multiplier = 0.01

    if to_ccy == 'GBX':
        if from_ccy == 'GBP':
            return 100.
        else:
            to_ccy = 'GBP'
            multiplier = 100.

    # Convert currency
    ccy_pair = from_ccy + to_ccy
    fx_data = PriceData.from_instrument_id(ccy_pair)
    fx_df = fx_data.df
    fx_df = fx_df.rename(columns={'close': ccy_pair})

    # Multiply for pounds/pence if necessary
    if multiplier:
        fx_df *= multiplier

    return fx_df


def convert_currency(price_df, fx_rate):
    """
    Convert a price time series using the given FX rate.

    Handles missing values, reshaping etc.

    Parameters
    ----------
    price_df: pd.DataFrame
        Price time series.
    fx_rate: pd.DataFrame or float
        Time series of FX rates.

    Returns
    -------
    converted_df: pd.DataFrame
        Time series of prices converted using the given FX rate.
    """
    if isinstance(fx_rate, pd.DataFrame):
        check_nonempty_dataframe(fx_rate, 'fx_rate')
        assert fx_rate.shape[1] == 1
        fx_reindexed = fx_rate.reindex(price_df.index)
        fx_reindexed.columns = price_df.columns
        fx_reindexed = fx_reindexed.ffill(limit=MAX_FFILL)  # ffill fx so we still have a signal

    elif isinstance(fx_rate, (int, float)):
        check_scalar_is_above_min_threshold(fx_rate, 'fx_rate', 0)
        fx_reindexed = fx_rate

    return price_df * fx_reindexed
