"""
Systematic framework.

Largely follows the methodology in Systematic Trading, Robert Carver.
"""

import numpy as np
import pandas as pd

from alphadog.internals.analytics import cross_sectional_mean
from alphadog.data.retrieval import PriceData
from alphadog.framework.config_handler import load_default_instrument_config
from alphadog.framework.constants import (
    AVG_FORECAST, MIN_FORECAST, MAX_FORECAST,
    MAX_DIVERSIFICATION_MULTIPLIER, VOL_TARGET
)
from alphadog.framework.signals_config import PARAMETERISED_STRATEGIES


class Portfolio:
    """
    Contains Subsystem objects for every instrument traded in the portfolio.

    Also calculate portfolio level statistics:
    - Portfolio returns
    - Correlations
    - p_weights and diversification multiplier - pass these down to Subsystem objects.
    - target_position
    - Required trades

    Take in a config yaml file of
    {'instrument_1': {'strategy_a': params, 'strategy_b': params}
    """
    def __init__(self, instrument_config=None, vol_target=VOL_TARGET):
        """
        Combine subsystems into a portfolio position for each instrument.

        Parameters
        ----------
        instrument_config: dict, optional
            Nested dict which specifies details of every instrument to run in the portfolio.
            If not supplied, defaults to the json config in the framework directory.
        vol_target: float
            The target annualised percentage volatility.
            If not supplied, defaults to the constant specified in the framework directory.
        """
        # TODO - handle the instrument's position in the hierarchy
        self.instrument_config = instrument_config or load_default_instrument_config()
        self.vol_target = vol_target

    def run_subsystems(self):
        """
        Run all subsystems supplied for the Instrument.

        Returns
        -------
        Sets forecast_list and capped_forecasts
        """
        instrument_ids = list(self.instrument_config.keys())
        subsystem_list = []
        for instrument_id in instrument_ids:
            try:
                instrument_dict = self.instrument_config[instrument_id]
                subsystem = Subsystem(instrument_dict)
                subsystem_list.append(subsystem)
            except:
                print(f"Failed on {instrument_id}")
        self.subsystem_list = subsystem_list

    def combine_subsystems(self):
        """
        Combine subsystems into a single target position for the Instrument.

        Returns
        -------
        """

        # Weight the instrument subsystems and scale to get target portfolio positions.
        diversification_multipliers = [
            get_diversification_multiplier(subsystem.subsystem_position, self.vol_target)
            for subsystem in self.subsystem_list
        ]
        self.diversification_multipliers = diversification_multipliers
        self.pweights = get_pweights(self.subsystem_list)

        # TODO: account for missing data, varying pweights
        weighted_subsystems = [
            subsystem.subsystem_position * weight * div_mult for subsystem, weight, div_mult
            in zip(self.subsystem_list, self.pweights, self.diversification_multipliers)
        ]

        self.target_position = pd.concat(weighted_subsystems, axis=1).sum(axis=1)


class Subsystem:
    """
    Takes an instrument configuration and calculates all Forecast objects and
    related parameters for that instrument.

    Combine forecasts and handle position sizing to give a subsystem position.

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
    def __init__(self, instrument_dict, vol_target=VOL_TARGET):
        """
        Initialise the Subsystem with the supplied forecasts for the instrument

        Parameters
        ----------
        instrument_dict: dict
            A single instrument from the `instrument_config.json`
        vol_target: float
            The target annualised percentage volatility.
        """
        self.instrument_id = instrument_dict['instrument_id']
        self.is_traded = instrument_dict['is_traded']
        self.currency = instrument_dict['currency']
        self.signals = instrument_dict['signals'] if self.is_traded else []
        self.vol_target = vol_target
        self.price_data = PriceData.from_instrument_id(self.instrument_id)

        # TODO - handle the signal's position in the hierarchy
        self.run_forecasts()
        self.combine_forecasts()

    def run_forecasts(self):
        """
        Run all forecasts supplied for the Instrument.

        Returns
        -------
        Sets forecast_list and capped_forecasts
        """

        forecast_list = []
        for signal_name in self.signals:
            signal_func = PARAMETERISED_STRATEGIES[signal_name]['signal_func']
            input_df = self.price_data.df  # TODO handle passing different data objects to different signals
            forecast = Forecast(signal_func,
                                {'price_df': input_df},
                                self.instrument_id,
                                f"{self.instrument_id}|{signal_name}")
            forecast_list.append(forecast)

        self.forecast_list = forecast_list
        self.capped_forecasts = [fc.capped_forecast for fc in forecast_list]

    def combine_forecasts(self):
        """
        Combine all signals in the subsystem and scale

        Returns
        -------

        """
        # Combine signals
        self.fweights = get_fweights(self.capped_forecasts)
        self.combined_forecasts = combine_signals(self.capped_forecasts, self.fweights, self.vol_target)  # noqa

        # Calc subsystem position
        self.vol_scalar = get_vol_scalar()  # TODO: implement this
        self.subsystem_position = self.vol_scalar * self.combined_forecasts / AVG_FORECAST


class Forecast:
    """
    Calculate a single strategy on a particular Instrument.

    Takes generic params which the signal function requires to run. This does not have to be
    a price timeseries; could be fundamental data, machine learned features/parameters.

    The signal function can then take this and return a timeseries of forecasts.
    """
    def __init__(self, signal_func, params, instrument_id='id', name='forecast'):
        """
        Initialise the forecast with a given function, parameters and optional name.

        Parameters
        ----------
        signal_func: function
            Function to run the signal
        params: dict
            Kwargs to pass to the function
        instrument_id: str, optional
            Identifier for the instrument
        name: str, optional
            Forecast name to identify this signal
        """
        self._signal_func = signal_func
        self._params = params
        self._instrument_id = instrument_id
        self._name = name

        self._raw_forecast = None
        self._forecast_scalar = None
        self._scaled_forecast = None
        self._capped_forecast = None
        self.run_signal()

    @property
    def signal_func(self):
        """function: Function to run the signal."""
        return self._signal_func

    @property
    def params(self):
        """dict: Parameters to run the signal with."""
        return self._params

    @property
    def instrument_id(self):
        """str: Instrument identifier."""
        return self._instrument_id

    @property
    def name(self):
        """str: Name of the forecast"""
        return self._name

    @property
    def raw_forecast(self):
        """pd.DataFrame: The raw output of running the signal_func with the given parameters."""
        return self._raw_forecast

    @property
    def forecast_scalar(self):
        """
        pd.Series
            The multiplier required to scale the raw forecast to have the correct mean value.
            Returns a float for each column of the raw forecast.
        """
        return self._forecast_scalar

    @property
    def scaled_forecast(self):
        """pd.DataFrame: The raw forecast scaled to have the correct mean value."""
        return self._scaled_forecast

    @property
    def capped_forecast(self):
        """
        pd.DataFrame
            The scaled forecast capped at minimum/maximum values.

            We also combine multicolumn signals by taking the cross-sectional mean.
            This is equivalent to treating them as separate signals with equal Sharpe ratios
            and equal pairwise correlations.
        """
        return self._capped_forecast

    def run_signal(self):
        """
        Run the signal with the specified params.

        Populates values for the raw, scaled and capped forecasts.
        """
        self._raw_forecast = self.signal_func(**self.params)
        self._forecast_scalar = AVG_FORECAST / self.raw_forecast.abs().mean()
        self._scaled_forecast = self.raw_forecast * self.forecast_scalar

        # We may want to revisit this cross_sectional_mean step in future.
        capped_forecast = self.scaled_forecast.clip(lower=MIN_FORECAST, upper=MAX_FORECAST)
        capped_forecast = cross_sectional_mean(capped_forecast, self.name)
        self._capped_forecast = capped_forecast


###################
# Portfolio utils #
###################


##################
# Forecast utils #
##################

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
    subsystem_list: list(Subsysytem)
        The subsystems to include in a portfolio.

    Returns
    -------
    pweights: list(float)
        The fraction of the portfolio to allocate to each Subsystem in the list.
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

    signal_vol = signal_df.std().values[0]
    div_multiplier = min(target_vol / signal_vol, MAX_DIVERSIFICATION_MULTIPLIER)

    return div_multiplier


def combine_signals(signals, weights, target_vol):
    """
    Combine the input signals as the weighted sum, and scale the result to reach the `target_vol`.

    Parameters
    ----------
    signals: list(pd.DataFrame)
        List of Forecast signals or Subsystem signals which we want to combine.
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
