"""
Systematic framework.

Largely follows the methodology in Systematic Trading, Robert Carver.
"""


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


class Instrument:
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

    Takes a generic data object which is whatever data the forecast uses. This does not have to be
    a price timeseries; could be fundamental data, machine learned features/parameters.

    The signal function can then take this and return a timeseries o

    Contains signals and related parameters:
    - signal function and parameters (Strategy)
    - raw_forecast
    - scaled_forecast
    - capped_forecast
    - f_weight? Don't thinks so, but we'll see
    """
    def __init__(self, data):
        pass
