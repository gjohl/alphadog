from datetime import datetime

from alphadog.internals.exceptions import InputDataError
from alphadog.data.constants import OHLCV_COLS


######################
# Generic data utils #
######################

def staleness(input_df):
    """Returns the number of days since the last update.

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
    today = datetime.today()
    most_recent_date = input_df.index.max()
    date_diff = today - most_recent_date
    staleness_days = date_diff.days

    return staleness_days


def check_nonempty_dataframe(input_df, name):
    """
    Run data quality checks for generic input data.

    Parameters
    ----------
    input_df: pd.DataFrame
       Input data.
    name: str, optional
       Name to identify object. Used to log errors.

    Returns
    -------
    Return True if all tests pass, otherwise raise the appropriate error.
    """
    if input_df.dropna(how='all').shape[0] == 0:
        raise InputDataError(f"No data for {name}")

    return True


def check_scalar_is_above_min_threshold(input_scalar, input_name, threshold):
    """
    Check if a scalar is above a given threshold value. Raise if it is below.

    Parameters
    ----------
    input_scalar: int, float
        Variable to check.
    input_name: str
        Name of variable. Used for logging.
    threshold: int, float
        Threshold value that the input_scalar must be greater than or equal to.

    Returns
    -------
    None

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

    Returns
    -------
    Return True if all tests pass, otherwise raise the appropriate error.
    """
    # Check expected columns
    # TODO: should we allow additional columns?
    if not set(input_df.columns).issuperset(OHLCV_COLS):
        raise InputDataError(f"Incorrect columns for {name}")

    # Check valid prices - may need to rethink with futures prices which may go negative
    if not all(input_df >= 0):
        raise InputDataError(f"Found negative prices")

    return True
