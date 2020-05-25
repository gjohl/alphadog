from datetime import datetime

import pandas as pd

from alphadog.internals.exceptions import InputDataError
from alphadog.data.constants import PRICE_COLS


######################
# Generic data utils #
######################

def staleness(input_df):
    """
    Returns the number of days since the last update.

    Measures how up-to-date the data is.

    Parameters
    ----------
    input_df: pd.DataFrame
       Input data to check.

    Returns
    -------
    staleness_days: int
    Number of periods (days) since the last update in this data.
    """
    if input_df.empty:
        raise InputDataError("Cannot calculate staleness for an empty Dataframe.")
    if not isinstance(input_df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a datetime.")

    today = datetime.today()
    most_recent_date = input_df.index.max()
    date_diff = today - most_recent_date
    staleness_days = date_diff.days

    return staleness_days


def check_nonempty_dataframe(input_df, name):
    """
    Check the input DataFrame is populated.

    Parameters
    ----------
    input_df: pd.DataFrame
       Input data.
    name: str, optional
       Name to identify object. Used to log errors.

    Raises
    ------
    InputDataError
        Raises if the DataFrame is empty or all NaN.
    """
    if input_df.dropna(how='all').shape[0] == 0:
        raise InputDataError(f"No data for {name}")


def check_scalar_is_above_min_threshold(input_scalar, input_name, threshold):
    """
    Check if a scalar is greater than or equal to a given threshold value.

    Raise if the input is below the given threshold.

    Parameters
    ----------
    input_scalar: int, float
        Variable to check.
    input_name: str
        Name of variable. Used for logging.
    threshold: int, float
        Threshold value that the input_scalar must be greater than or equal to.

    Raises
    ------
    InputDataError
        Raises if input_scalar is below the given threshold
    """
    if input_scalar < threshold:
        raise InputDataError(f"Input {input_name} is below the threshold value {threshold}."
                             f" Got {input_scalar}")


########################
# Price-specific utils #
########################

def check_price_data(input_df, name):
    """
    Checks specific to price data.

    Here we check:
    - expected columns
    - non-negative values

    Parameters
    ----------
    input_df: pd.DataFrame
        Input data.
    name: str, optional
        Name to identify object. Used to log errors.

    Raises
    ------
    InputDataError
        Raises if the DataFrame has negative values or incorrect columns
    """
    check_nonempty_dataframe(input_df, name)

    # TODO: make these generic tests and fix negative price failures
    # Check expected columns
    if not set(input_df.columns).issuperset(PRICE_COLS):
        raise InputDataError(f"Incorrect columns for {name}")

    # Check valid prices - may need to rethink with futures prices which may go negative
    # if any(input_df < 0):
    #     raise InputDataError(f"Found negative prices for {name}")

