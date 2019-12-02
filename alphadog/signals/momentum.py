# import pandas as pd

from ..helpers.exceptions import ParameterError


def momentum_signal(df, fast, slow):
    """
    Returns a basic momentum signal as a volatility-scaled EWMA crossover.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to run momentum signal on
    fast: int
        Span of fast EWMA to use for momentum signal.
    slow: int
        Span of slow EWMA to use for momentum signal and vol scaling.

    Returns
    -------
    pd.DataFrame
        Columnwise momentum signal run over the input DataFrame.
        Returns the same columns as the input.

    Raises
    ------
    ParameterError
        Raises if the provided fast ewma span is greater than the slow span.
    """
    if fast > slow:
        raise ParameterError(f"Span of fast must be less than slow. Got fast={fast}, slow={slow}")

    fast_ewma = df.ewm(span=fast).mean()
    slow_ewma = df.ewm(span=slow).mean()
    ewvol = df.ewm(span=slow).std()
    return (fast_ewma - slow_ewma) / ewvol


def carry_signal():
    """
    Carry signal according to Carry (2012), Koijen, Moskowitz.

    Parameters
    ----------

    Returns
    -------

    """
    # TODO
    pass
