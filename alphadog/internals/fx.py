"""
Functionality to handle FX conversions.
"""

import pandas as pd

from alphadog.data.retrieval import PriceData


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

    # Handle no conversion
    if from_ccy == to_ccy:
        return 1.

    # Convert currency
    ccy_pair = from_ccy + to_ccy
    fx_data = PriceData.from_instrument_id(ccy_pair)
    fx_df = fx_data.df

    # Multiply for pounds/pence if necessary
    if multiplier:
        fx_df *= multiplier

    return fx_df
