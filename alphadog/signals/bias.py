from ..framework.constants import AVG_FORECAST


def long_bias_signal(price_df):
    """
    A signal which is a constant long signal for all of time.

    The value is set to the average forecast value for all timestamps.

    Parameters
    ----------
    price_df: pd.DataFrame
        DataFrame to run long bias signal on.

    Returns
    -------
    long_bias_df: pd.DataFrame
        Columnwise long bias signal.
        Returns the same columns as the input.
    """
    long_bias_df = price_df.copy()

    # Set all values to AVG_FORECAST while preserving dtypes
    for col in long_bias_df.columns:
        long_bias_df[col].values[:] = AVG_FORECAST

    return long_bias_df


def short_bias_signal(price_df):
    """
    A signal which is a constant short signal for all of time.

    The value is set to the negative of the average forecast value for all timestamps.

    Parameters
    ----------
    price_df: pd.DataFrame
        DataFrame to run short bias signal on.

    Returns
    -------
    short_bias_df: pd.DataFrame
        Columnwise short bias signal.
        Returns the same columns as the input.
    """
    short_bias_df = price_df.copy()

    # Set all values to negative AVG_FORECAST while preserving dtypes
    for col in short_bias_df.columns:
        short_bias_df[col].values[:] = -AVG_FORECAST

    return short_bias_df
