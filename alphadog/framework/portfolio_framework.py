"""
Systematic framework.

Largely follows the methodology in Systematic Trading, Robert Carver.
"""
import numpy as np
import pandas as pd

from .constants import (
    AVG_FORECAST, MIN_FORECAST, MAX_FORECAST,
    MAX_DIVERSIFICATION_MULTIPLIER, VOL_TARGET
)


class Portfolio:
    """
    Contains InstrumentForecast objects for every instrument traded in the portfolio.

    Also calculate portfolio level statistics:
    - Portfolio returns
    - Correlations
    - p_weights and diversification multiplier - pass these down to Instrument objects.
    - target_position
    - Required trades

    Take in a config yaml file of
    {'instrument_1': {'strategy_a': params, 'strategy_b': params}
    """
    def __init__(self, instrument_forecasts, vol_target=VOL_TARGET):
        """
        Combine instrument forecasts into a portfolio position for each instrument

        Parameters
        ----------
        instrument_forecasts: list(InstrumentForecast)
        """
        self.instrument_forecasts = instrument_forecasts
        self.vol_target = vol_target

        # Weight the instrument subsystems and scale to get target portfolio positions.
        diversification_multipliers = [get_diversification_multiplier(subsystem, self.vol_target)
                                       for subsystem in self.instrument_forecasts]

        self.pweights = get_pweights(self.instrument_forecasts)
        self.diversification_multipliers = diversification_multipliers

        instrument_positions = [
            subsystem * weight * div_mult for subsystem, weight, div_mult
            in zip(self.instrument_forecasts, self.pweights, self.diversification_multipliers)
        ]

        self.historical_instrument_positions = pd.concat(instrument_positions, axis=1)

        # TODO: finish this class according to the readme



class InstrumentForecast:
    """
    Contains all strategies (Forecast objects) and related parameters for a particular instrument.

    Combine forecasts and handle position sizing is handled here to give a subsystem position.

    Contain/calculate:
    - p_weights
    - diversification multiplier
    - block value
    - price vol
    - fx
    - vol_target
    - f_weight - pass these down to forecast objects?
    - subsystem_position
    - instrument_position (scale and cap subsystem)
    - target_position (round)
    """
    def __init__(self, forecast_list, vol_target=VOL_TARGET):
        """
        Initialise the InstrumentForecast with the supplied forecasts for the instrument

        Parameters
        ----------
        forecast_list: list(Forecast)
            The Forecast signals to run on an instrument.
        """
        self.forecast_list = forecast_list
        self.vol_target = vol_target

        # Combine signals
        self.fweights = get_fweights(self.forecast_list)
        self.combined_forecasts = combine_signals(self.forecast_list, self.fweights, self.vol_target)  # noqa

        # Calc subsystem position
        self.vol_scalar = get_vol_scalar()  # TODO: implement this
        self.subsystem_position = self.vol_scalar * self.combined_forecasts / AVG_FORECAST



class Forecast:
    """
    Calculate a single strategy on a particular Instrument.

    Takes generic params which is signal function requires to run. This does not have to be
    a price timeseries; could be fundamental data, machine learned features/parameters.

    The signal function can then take this and return a timeseries of forecasts.
    """
    def __init__(self, signal_func, params, instrument_id, name):
        """
        Initialise the forecast with a given function, parameters and optional name.

        Parameters
        ----------
        signal_func: function
            Function to run the signal
        params: dict
            Kwargs to pass to the function
        instrument_id: str
            Identifier for the instrument
        name: str
            Forecast name to identify this signal
        """
        self.signal_func = signal_func
        self.params = params
        self.instrument_id = instrument_id
        self.name = name

        self.run_signal()

    def run_signal(self):
        """
        Run the signal with the specified params

        Returns
        -------
        Sets the raw, scaled and capped forecast attributes
        """
        self.raw_forecast = self.signal_func(**self.params)
        self.forecast_scalar = AVG_FORECAST / self.raw_forecast.mean()
        self.scaled_forecast = self.raw_forecast * self.forecast_scalar
        self.capped_forecast = self.scaled_forecast.clip(lower=MIN_FORECAST, upper=MAX_FORECAST)


def get_fweights(forecast_list):
    """
    Return a list of forecast weights for a given forecast_list.

    Parameters
    ----------
    forecast_list: list(Forecast)
        The Forecast signals to include in an instrument.

    Returns
    -------
    fweights: list(float)
        The fraction of the instrument to allocate to each Forecast in the list.
    """
    # TODO: As a first pass, this currently just assigns equal weights to all forecasts.
    #  These should actually vary depending on correlations.
    num_forecasts = len(forecast_list)
    equal_weight = 1 / num_forecasts
    fweights = [equal_weight] * num_forecasts

    return fweights


def get_pweights(subsystem_list):
    """
    Return a list of portfolio weights for a given subsystem list

    Parameters
    ----------
    subsystem_list: list(InstrumentForecast)
        The instrument forecasts to include in a portfolio.

    Returns
    -------
    pweights: list(float)
        The fraction of the portfolio to allocate to each InstrumentForecast subsystem in the list.
    """
    # TODO: As a first pass this just assigns equal weights. Calculate this correctly.
    #   It may make sense to combine the logic for fweights and pweights into a single function
    #   and call that here and in get_fweights.
    return get_fweights(subsystem_list)


def get_diversification_multiplier(signal_df, target_vol):
    """
    Return the diversification multiplier for a combined signal

    Parameters
    ----------
    signal_df: pd.DataFrame
        A single column DataFrame which is typically the result of combining other signals.
    target_vol: float
        The target annualised percentage volatility.

    Returns
    -------
    div_multiplier: float
        The scaling factor to scale the input `signal_df` by to make it achieve the `target_vol`.
    """
    # TODO: As a first pass, just scale by the volatility of however much data we have.
    #  Perhaps we may want to try a rolling window and/or ewmvol?
    assert target_vol > 0

    signal_vol = signal_df.std()
    div_multiplier = min(target_vol / signal_vol, MAX_DIVERSIFICATION_MULTIPLIER)

    return div_multiplier


def combine_signals(signals, weights, target_vol):
    """
    Combine the input signals as the weighted sum, and scale the result to reach the `target_vol`.

    Parameters
    ----------
    signals: list(pd.DataFrame)
        List of Forecast signals or Instrument signals which we want to combine.
        Each signal should be a single column DataFrame.
    weights: list(float)
        Weight to assign each signal in the combined result.
    target_vol: float
        The target annualised percentage volatility.

    Returns
    -------
    combined_df: pd.DataFrame
        A single column DataFrame which is the weighted sum of input DataFrames, scaled to
        achieve the vol target.
    """
    # Check the weights are valid
    assert len(signals) == len(weights)
    assert np.isclose(sum(weights), 1)

    # Combine signals according to their weights
    weighted_forecasts = sum(x * y for x, y in zip(signals, weights))

    # Scale back up to vol target
    div_multiplier = get_diversification_multiplier(weighted_forecasts, target_vol)
    combined_df = weighted_forecasts * div_multiplier

    # TODO: also cap here?

    return combined_df


def get_vol_scalar():
    """
    Calculate the volatility scalar for a given instrument.

    Parameters
    ----------
    block_value
    price_vol
    fx_rate
    target_vol

    Returns
    -------
    vol_scalar: float
        The volatility scalar for a given instrument.
    """
    # TODO: This is not yet implemented.
    return 1.
