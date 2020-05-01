import numpy as np
# import pandas as pd

from ..internals.exceptions import ParameterError
from ..framework.constants import MAX_FORECAST, MIN_FORECAST


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
    momentum_df: pd.DataFrame
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

    # Reinstate NaNs where they were in the input DataFrame
    momentum_df = (fast_ewma - slow_ewma) / ewvol
    momentum_df[df.isna()] = np.nan

    return momentum_df


def breakout_signal(df, lookback_period, smooth_period=None):
    """
    A continuous breakout signal that indicates when the data is nearing an extreme relative
    to its recent history (rolling min or max).

    Similar in concept to a stochastic oscillator or Donchian channel. Based on the idea discussed
    in https://qoppac.blogspot.com/2016/05/a-simple-breakout-trading-rule.html?m=1

    The breakout rule is defined as the difference of the current value to the midpoint, scaled
    by the rolling_range of values, i.e.:
    rolling_midpoint = (rolling_max + rolling_min) / 2
    raw_breakout = (price - rolling_midpoint) / (rolling_max - rolling_min)

    We then scale and smooth the forecast. Scaling is done here (as well as the usual point in the
    framework) because there is a natural range of 1 for the raw forecast:
    -0.5 (when price = roll_min) to +0.5 (price = roll_max).
    Thus, we can scale here to get an a priori scaled signal. We will still scale in the framework,
    but this allows us to sanity check the signal as those scaling factors should then be close
    to 1.
    scaled_breakout = max_forecast_range * raw_breakout

    We smooth the forecast to slow down it down to a sensible speed. The raw forecast inherits the
    speed of the current data, i.e. it moves with daily price. This is unreasonably fast since the
    breakout signal is intended to pick up signals on the horizon of its lookback speed.
    The smoothing is proportional to the lookback period unless otherwise specified.

    10 days seems a sane starting point, also exactly two weeks in business days.
    If I keep doubling I get 10,20,40,80,160 and 320 day lookbacks before we're getting a
    little too slow.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to run breakout signal on.
    lookback_period: int
        Number of days to use to calculate rolling extremes.
    smooth_period: int
        Span of EWMA smoothing to use to slow down the signals. Defaults to lookback_period / 4.

    Returns
    -------
    pd.DataFrame
        Columnwise breakout signal run over the input DataFrame.
        Returns the same columns as the input.

    Raises
    ------
    ParameterError
        Raises if the provided smooth_period is greater than the lookback_period.
    """
    if smooth_period is None:
        smooth_period = max(int(lookback_period / 4), 1)

    if smooth_period > lookback_period:
        raise ParameterError(f"Lookback period {lookback_period} must be greater than "
                             f"smooth period {smooth_period}")

    # Calculating rolling stats
    min_lookback_periods = int(min(df.shape[0], np.ceil(lookback_period / 2)))
    min_smooth_periods = int(np.ceil(smooth_period / 2))

    rolling_max = df.rolling(lookback_period, min_periods=min_lookback_periods).max()
    rolling_min = df.rolling(lookback_period, min_periods=min_lookback_periods).min()
    rolling_midpoint = (rolling_max + rolling_min) / 2

    # Calculate breakout signal, scale by natural scaling and slow down
    max_forecast_range = MAX_FORECAST - MIN_FORECAST
    raw_breakout = (df - rolling_midpoint) / (rolling_max - rolling_min)
    scaled_breakout = max_forecast_range * raw_breakout
    smoothed_breakout = (scaled_breakout
                         .ewm(span=smooth_period, min_periods=min_smooth_periods)
                         .mean())
    # Reinstate NaNs where they were in the input DataFrame
    smoothed_breakout[df.isna()] = np.nan

    return smoothed_breakout
