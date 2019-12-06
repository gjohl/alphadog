"""
Systematic framework.

Largely follows the methodology in Systematic Trading, Robert Carver.
"""
from .constants import AVG_FORECAST, MIN_FORECAST, MAX_FORECAST


class Portfolio:
    """
    Contains Instrument objects for every instrument traded in the portfolio.

    Also calculate portfolio level statistics:
    - Portfolio returns
    - Correlations
    - p_weights and diversification multiplier - pass these down to Instrument objects.
    - target_position
    - Required trades

    Take in a config yaml file of
    {'instrument_1': {'strategy_a': params, 'strategy_b': params}
    """
    def __init__(self):
        pass


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
    - vol target
    - f_weight - pass these down to forecast objects?
    - subsystem_position
    - instrument_position (scale and cap subsystem)
    - target_position (round)
    """
    def __init__(self, price, fx_rate, config):
        pass


class Forecast:
    """
    Calculate a single strategy on a particular Instrument.

    Takes generic params which is signal function requires to run. This does not have to be
    a price timeseries; could be fundamental data, machine learned features/parameters.

    The signal function can then take this and return a timeseries of forecasts.

    Contains signals and related parameters:
    - signal function and parameters (Strategy)
    - raw_forecast
    - scaled_forecast
    - capped_forecast
    - f_weight? Don't thinks so, but we'll see
    """
    def __init__(self, signal_func, params, instrument_id, name):
        """
        Initialise the forecast with a given function, parameters and optional name.

        Parameters
        ----------
        signal_func: function
            Function to run fo the signal
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

        # Run the strategy
        self.raw_forecast = self.signal_func(**self.params)
        self.forecast_scalar = AVG_FORECAST / self.raw_forecast.mean()
        self.scaled_forecast = self.raw_forecast * self.forecast_scalar
        self.capped_forecast = self.scaled_forecast.clip(lower=MIN_FORECAST, upper=MAX_FORECAST)
