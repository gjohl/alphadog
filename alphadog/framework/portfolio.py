"""
Systematic framework.

Largely follows the methodology in Systematic Trading, Robert Carver.
"""
import logging

import numpy as np
import pandas as pd

from alphadog.analytics.returns import returns, cross_sectional_mean
from alphadog.internals.exceptions import InputDataError, DimensionMismatchError
from alphadog.internals.fx import get_fx, convert_currency
from alphadog.data.data_quality import (
    check_scalar_is_above_min_threshold, check_nonempty_dataframe
)
from alphadog.data.retrieval import PriceData
from alphadog.framework.config_handler import (
    load_default_instrument_config, Instrument
)
from alphadog.framework.constants import (
    AVG_FORECAST, MIN_FORECAST, MAX_FORECAST,
    MAX_DIVERSIFICATION_MULTIPLIER, VOL_TARGET, VOL_SPAN,
    TRADING_DAYS_PER_YEAR, PORTFOLIO_CCY,
    STARTING_CAPITAL
)
from alphadog.framework.signals_config import DATA_FIXTURES
from alphadog.framework.weights import STRATEGY_WEIGHTS, INSTRUMENT_WEIGHTS


class Portfolio:
    """
    Takes an instrument configuration and computes the Subsystem for every Instrument.

    For each instrument, calculate associated portfolio parameters
    (portfolio weights, diversification multipliers), and use these to combine
    Subsystems to get a target position for each traded Instrument.

    TODO: Also calculate portfolio level statistics:
    - Portfolio returns
    - Correlations
    - Required trades
    """
    def __init__(self, instrument_config=None, vol_target=VOL_TARGET):
        """
        Initialise the Portfolio with the supplied instruments.

        Parameters
        ----------
        instrument_config: dict, optional
            Nested dict which specifies details of every instrument to run in the portfolio.
            If not supplied, defaults to the json config in the framework directory.
        vol_target: float
            The target annualised percentage volatility.
            If not supplied, defaults to the constant specified in the framework directory.
        """
        self._instrument_config = instrument_config or load_default_instrument_config()
        self._vol_target = vol_target

        self._subsystems = None
        self._diversification_multiplier = None
        self._pweights = None
        self._target_position = None
        self._target_notional = None

    def __repr__(self):
        return f"Portfolio {self.__class__}\nContaining instruments: {self.traded_instruments}"

    @property
    def instrument_config(self):
        """
        dict:
            Nested dictionary which specifies details of every instrument to run in the portfolio.
        """
        return self._instrument_config

    @property
    def vol_target(self):
        """float: The target annualised percentage volatility."""
        return self._vol_target

    @property
    def subsystems(self):
        """list(Subsystem): All subsystems in the portfolio."""
        return self._subsystems

    @property
    def diversification_multiplier(self):
        """float: Instrument diversification multiplier."""
        return self._diversification_multiplier

    @property
    def pweights(self):
        """list(float): Portfolio weights for each subsystem."""
        return self._pweights

    @property
    def target_position(self):
        """pd.DataFrame: Target position for each instrument in the portfolio."""
        return self._target_position

    @property
    def target_notional(self):
        """pd.DataFrame: Target GBP position for each instrument in the portfolio."""
        return self._target_notional

    @property
    def traded_instruments(self):
        """ list(str): instrument_names of all traded instruments."""
        return [inst_id for inst_id
                in self.instrument_config.keys()
                if self.instrument_config[inst_id]['is_traded']]

    @property
    def instruments(self):
        """
        All Instruments in the portfolio.

        Returns
        -------
        dict(Instrument)
            instrument_name: instrument_object
        """
        return {inst_id: Instrument.from_config(self.instrument_config[inst_id])
                for inst_id in self.traded_instruments}

    @property
    def instrument_weights(self):
        """
        Calculate the weight associated with each instrument.

        Currently uses hard-coded weights from config.

        Returns
        -------
        dict:
            Return the weight for each instrument in the format
            {instrument_id: weight}
        """
        weights_dict = {}
        for instrument_id, instrument in self.instruments.items():
            weight = get_weights_from_config(instrument, INSTRUMENT_WEIGHTS)
            weights_dict[instrument_id] = weight

        return weights_dict

    def run_subsystems(self, rescale=True):
        """
        Run all subsystems supplied for the Instrument.

        Parameters
        ----------
        rescale: bool
            Whether to scale the pweights if they don't already sum to 1.
            Default True

        Returns
        -------
        None
            Sets subsystems.
            Runs each subsystem in turn and creates a list of Subsystems and a list of
            their corresponding weights.
        """
        subsystem_list = []
        pweight_list = []

        for instrument_id, instrument in self.instruments.items():
            subsystem_list.append(Subsystem(instrument))
            pweight_list.append(self.instrument_weights[instrument_id])

        total_weight = sum(pweight_list)
        if rescale and not np.isclose(total_weight, 1):
            logging.warning(f"Rescaling pweights - original total weight was {total_weight}")
            rescale_factor = 1 / total_weight
            pweight_list = [pw * rescale_factor for pw in pweight_list]

        self._subsystems = subsystem_list
        self._pweights = pweight_list

    def combine_subsystems(self):
        """
        Combine subsystems into a single target position for the Instrument.

        Returns
        -------
        """
        subsystem_returns = [sub.subsystem_returns for sub in self.subsystems]
        div_multiplier = get_diversification_multiplier(subsystem_returns, self.pweights)
        self._diversification_multiplier = div_multiplier

        # TODO: account for missing data, varying pweights
        # Positions in number of shares
        subsystem_positions = [sub.subsystem_position for sub in self.subsystems]
        portfolio_positions_list = [x * y for x, y in zip(subsystem_positions, self.pweights)]
        portfolio_positions = pd.concat(portfolio_positions_list, axis=1)
        self._target_position = portfolio_positions * self.diversification_multiplier

        # Positions in notional GBP
        subsystem_prices = [sub.converted_price for sub in self.subsystems]
        subsystem_cols = [sub.instrument_id for sub in self.subsystems]
        prices = pd.concat(subsystem_prices, axis=1)
        prices.columns = subsystem_cols
        self._target_notional = self._target_position * prices


class Subsystem:
    """
    Takes an Instrument object configuration and calculates all Forecast objects and
    related parameters for that instrument.

    Combine forecasts and handle position sizing to give a subsystem position.

    Calculate:
    - forecasts
    - f_weights
    - vol scalar - price, fx, vol_target, trading_capital
    - subsystem_position
    """
    def __init__(self, instrument, vol_target=VOL_TARGET):
        """
        Initialise the Subsystem with the supplied forecasts for the instrument.

        Parameters
        ----------
        instrument: Instrument
            A single instrument to run.
        vol_target: float
            The target annualised percentage volatility.
        """
        self._instrument = instrument
        self._vol_target = vol_target

        self.data = self.load_data_fixtures()
        self.price_data = PriceData.from_instrument_id(self.instrument_id)

        self._forecast_list = None
        self._capped_forecasts = None
        self._fweights = None
        self._combined_forecast = None
        self._vol_scalar = None
        self._subsystem_position = None
        self._subsystem_returns = None

        self.run_forecasts()
        self.calc_position()
        self.calc_returns()

    def __repr__(self):
        return f"{self.instrument_id} Subsystem {self.__class__}"

    @property
    def instrument(self):
        """Instrument: The Instrument object to run the Subsystem for."""
        return self._instrument

    @property
    def vol_target(self):
        """float: The target annualised percentage volatility."""
        return self._vol_target

    @property
    def forecast_list(self):
        """list(Forecast): List of Forecasts for the strategies run on this instrument."""
        return self._forecast_list

    @property
    def capped_forecasts(self):
        """list(pd.DataFrame): List of capped forecast DataFrames, one per strategy."""
        return self._capped_forecasts

    @property
    def fweights(self):
        """list(float): Forecast weights."""
        return self._fweights

    @property
    def combined_forecast(self):
        """pd.DataFrame: Forecasts weighted and vol scaled."""
        return self._combined_forecast

    @property
    def vol_scalar(self):
        """
        pd.DataFrame:
            Volatility scalar.
            The number of instrument blocks required to hit the volatility target
            with an average forecast.
        """
        return self._vol_scalar

    @property
    def subsystem_position(self):
        """pd.DataFrame: Target position to take in this instrument."""
        return self._subsystem_position

    @property
    def subsystem_returns(self):
        """pd.DataFrame: The returns of the Subsystem. Single col with instrument_id."""
        return self._subsystem_returns

    @property
    def currency(self):
        """str: Currency ISO code."""
        return self.instrument.currency

    @property
    def fx_rate(self):
        """
        pd.DataFrame, float:
            The multiplicative FX rate to convert this instrument to the portfolio currency.
        """
        return get_fx(self.currency, PORTFOLIO_CCY)

    @property
    def converted_price(self):
        """pd.DataFrame: The instrument price converted to the portfolio currency.
        """
        return convert_currency(self.price_data.df, self.fx_rate)

    @property
    def instrument_id(self):
        """str: The instrument identifier."""
        return self.instrument.instrument_id

    @property
    def is_traded(self):
        """bool: Flags whether this instrument is traded."""
        return self.instrument.is_traded

    @property
    def required_data_fixtures(self):
        """
        List of unique data fixtures required for the signals run on this object.

        Returns
        -------
        list(str)
            List of data fixture names.
        """
        return self.instrument.required_data_fixtures

    @property
    def strategies(self):
        """dict: Strategies to run for this instrument."""
        return self.instrument.strategies

    @property
    def strategy_weights(self):
        """
        Calculate the weight associated with each strategy.

        Currently uses hard-coded weights from config.

        Returns
        -------
        dict:
            Return the weight for each strategy in the format
            {strategy_name: weight}
        """
        weights_dict = {}
        for strat_name, strat in self.strategies.items():
            weight = get_weights_from_config(strat, STRATEGY_WEIGHTS)
            weights_dict[strat_name] = weight

        return weights_dict

    @property
    def trading_capital(self):
        """
        TODO: This is currently a constant.
         This should be a timeseries to reflect rolling up gains / rolling down losses.
        """
        return STARTING_CAPITAL

    def load_data_fixtures(self):
        """
        Load all data required for the strategies to be run on this instrument.

        Returns
        -------
        dict:
            All data fixtures required for this instrument, in the format:
            {fixture_name: fixture_df}
        """
        return {fixture: DATA_FIXTURES[fixture](self.instrument_id)
                for fixture
                in self.required_data_fixtures}

    def run_forecasts(self):
        """
        Run all forecasts supplied for the Instrument.

        Returns
        -------
        Sets the forecast_list and capped_forecasts properties of the Subsystem.
        """
        forecast_list = []
        fweight_list = []

        for strat_name, strat in self.strategies.items():
            signal_func = strat.signal_func
            params = {fixture: self.data[fixture] for fixture in strat.required_data_fixtures}
            forecast = Forecast(signal_func,
                                params,
                                self.instrument_id,
                                f"{self.instrument_id}|{strat.strategy_name}")
            forecast_list.append(forecast)
            fweight_list.append(self.strategy_weights[strat_name])

        self._fweights = fweight_list
        self._forecast_list = forecast_list
        self._capped_forecasts = [fc.capped_forecast for fc in forecast_list]

    def calc_position(self):
        """
        Combine all signals in the subsystem, combine and scale.

        Returns
        -------
        Sets the subsystem_position and associated parameters.
        """
        # Combine signals
        self._combined_forecast = combine_signals(self.capped_forecasts, self.fweights)

        # Calc subsystem position
        self._vol_scalar = get_vol_scalar(
            self.price_data.df, self.fx_rate, self.vol_target, self.trading_capital
        )
        final_position = self.vol_scalar['vol_scalar'] * self.combined_forecast['combined'] / AVG_FORECAST
        self._subsystem_position = final_position.to_frame(self.instrument_id)

    def calc_returns(self):
        """
        Calculate the returns of the Subsystem.

        Calculate the price returns and use the subsystem positions to get the subsystem returns.

        Returns
        -------
        Sets the subsystem_returns parameter
        """
        instrument_returns = returns(self.price_data.df, 'geometric', percent=True).iloc[:, 0]
        positions = self.subsystem_position[self.instrument_id]
        subsystem_returns = instrument_returns.shift() * positions
        subsystem_returns = subsystem_returns.to_frame(self.instrument_id)
        self._subsystem_returns = subsystem_returns


class Forecast:
    """
    Calculate a single strategy on a particular Instrument.

    Takes generic params which the signal function requires to run. This does not have to be
    a price timeseries; could be fundamental data, machine learned features/parameters.

    The signal function can then take this and return a timeseries of raw forecasts.
    These are then scaled and capped to have normalised absolute mean, min and max values.
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

    def __repr__(self):
        return f"{self.name} Forecast {self.__class__}"

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
        if self.raw_forecast.empty:
            raise InputDataError(f"{self.name} has an empty raw forecast. Check inputs.")
        self._forecast_scalar = AVG_FORECAST / self.raw_forecast.abs().median()
        self._scaled_forecast = self.raw_forecast * self.forecast_scalar

        # We may want to revisit this cross_sectional_mean step in future.
        capped_forecast = self.scaled_forecast.clip(lower=MIN_FORECAST, upper=MAX_FORECAST)
        capped_forecast = cross_sectional_mean(capped_forecast, self.name)
        self._capped_forecast = capped_forecast


#########
# Utils #
#########

def combine_signals(signals, weights):
    """
    Combine the input signals as the weighted sum, and scale the result to reach the `target_vol`.

    Parameters
    ----------
    signals: list(pd.DataFrame)
        List of Forecast signals or Subsystem signals which we want to combine.
        Each signal should be a single column DataFrame.
    weights: list(float)
        Weight to assign each signal in the combined result.

    Returns
    -------
    combined_df: pd.DataFrame
        A single column DataFrame which is the weighted sum of input DataFrames, scaled to
        achieve the vol target.
    """
    # Check the weights are valid
    if len(signals) != len(weights):
        raise DimensionMismatchError(f"Number of weights does not equal number of signals. "
                                     f"Got {len(weights)} weights but {len(signals)} signals.")
    if not np.isclose(sum(weights), 1):
        raise InputDataError("Weights must sum to 1.")

    # Combine signals according to their weights
    weighted_forecasts_list = [x * y for x, y in zip(signals, weights)]
    weighted_forecasts = pd.concat(weighted_forecasts_list, axis=1).sum(axis=1)

    # Scale back up to vol target
    div_multiplier = get_diversification_multiplier(signals, weights)
    combined_df = weighted_forecasts * div_multiplier
    combined_df = combined_df.clip(lower=MIN_FORECAST, upper=MAX_FORECAST)
    combined_df = combined_df.to_frame('combined')

    return combined_df


def get_diversification_multiplier(signals, weights):
    """
    Return the diversification multiplier for a combined signal.

    Use the weights and correlations of the signals to calculate the reduction in volatility
    when combining signals as sqrt(WHWt). The diversification multiplier is the reciprocal of this,
    used to scale the volatility of the combined signal back up.

    Parameters
    ----------
    signals: list(pd.DataFrame)
        List of Forecast signals or Subsystem signals which we want to combine.
        Each signal should be a single column DataFrame.
    weights: list(float)
        Weight to assign each signal in the combined result.

    Returns
    -------
    div_multiplier: float
        The scaling factor to scale the weighted combination of input signals to make it
        achieve the target volatility.
    """
    # Sanitise inputs
    if len(signals) != len(weights):
        raise DimensionMismatchError(f"Number of weights does not equal number of signals. "
                                     f"Got {len(weights)} weights but {len(signals)} signals.")
    if not np.isclose(sum(weights), 1):
        raise InputDataError("Weights must sum to 1.")
    df = pd.concat(signals, axis=1)

    # Calc correlation matrix H and floor
    corr_matrix = df.corr()
    corr_matrix = corr_matrix.fillna(0)
    corr_matrix = corr_matrix.clip(lower=0)

    # WHWt
    weights_array = np.array(weights)
    corr_array = corr_matrix.values.copy()
    np.fill_diagonal(corr_array, 1.)  # Needed due to earlier fillna
    combined_corr = np.matmul(weights_array, corr_array)
    combined_corr = np.matmul(combined_corr, weights_array.T)

    # 1 / sqrt(WHWt)
    div_multiplier = 1 / np.sqrt(combined_corr)

    # Clip between 1 and MAX_DIVERSIFICATION_MULTIPLIER
    div_multiplier = max(div_multiplier, 1)
    div_multiplier = min(div_multiplier, MAX_DIVERSIFICATION_MULTIPLIER)

    return div_multiplier


def get_weights_from_config(config_object, weights_config):
    """
    Get the weight for a given Strategy/Instrument object using the weights in a given config.

    Gets the weight for each level, assigns equal weights to the lowest level,
    and multiplies these to give the weight for this specific object.

    Parameters
    ----------
    config_object: Instrument, Strategy
        The instrument or strategy to calculate the weight for
    weights_config: dict
        Nested dict containing each level of the hierarchy as a key and the weight for
        an item at a particular level stored under the 'weight' key

    Returns
    -------
    float:
        Weight for the given `config_object`
    """
    # TODO: Tidy this function up. This is ugly but it works.
    depth = config_object.depth()

    # Get weight for each level in turn
    weights_by_level = []

    # Level 1
    level_1_name = config_object.hierarchy_1
    level_1_weight = weights_config[level_1_name]['weight']
    weights_by_level.append(level_1_weight)

    # Level 2
    if depth > 1:
        level_2_name = config_object.hierarchy_2
        level_2_weight = weights_config[level_1_name][level_2_name]['weight']
        weights_by_level.append(level_2_weight)

    # Level 3
    if depth > 2:
        level_3_name = config_object.hierarchy_3
        level_3_weight = weights_config[level_1_name][level_2_name][level_3_name]['weight']
        weights_by_level.append(level_3_weight)

    if depth > 3:
        raise NotImplementedError("Weights for depths greater than 3 not implemented yet.")

    # Siblings on lowest level
    num_siblings = len(config_object.siblings())
    lowest_level_weight = 1 / num_siblings if num_siblings > 0 else 1
    weights_by_level.append(lowest_level_weight)

    return np.prod(weights_by_level)


############################
# Portfolio-specific utils #
############################


############################
# Subsystem-specific utils #
############################

def get_vol_scalar(price_df, fx_rate, vol_target, trading_capital):
    """
    Calculate the volatility scalar for a given instrument.

    Instrument value volatility gives the vol contribution per block of the instrument.
    Ignoring forecasts for the moment, the vol_scalar gives the number of blocks to buy to achieve
    the vol target using this instrument alone.

    Parameters
    ----------
    price_df: pd.DataFrame
        Daily close price of the instrument being traded. This assumes equities only.
    fx_rate: pd.DataFrame or float
        Daily FX rate to GBP
    vol_target: float
        The target annualised percentage volatility.
    trading_capital: pd.DataFrame
        Timeseries of capital available to invest. GBP amount.
        TODO: This is currently a scalar starting amount which does not reflect
         rolling up/ rolling down losses.
         The current implementation is equivalent to adding capital following losses and
         removing gains, so this does not reflect any compounding

    Returns
    -------
    vol_scalar: pd.DataFrame
        The volatility scalar for a given instrument.
        This is the number of blocks of the given instrument required to hit the vol target.
    """
    instrument_value_vol = get_instrument_value_volatility(price_df, fx_rate)
    cash_vol_target_daily = get_cash_vol_target_daily(vol_target, trading_capital)
    vol_scalar = cash_vol_target_daily / instrument_value_vol
    vol_scalar = pd.DataFrame(vol_scalar)
    vol_scalar.columns = ['vol_scalar']

    return vol_scalar


def get_instrument_value_volatility(price_df, fx_rate, asset_class='equity'):
    """
    Get the instrument value volatility for an instrument.

    Instrument value volatility gives the vol contribution per block of the instrument
    in the portfolio currency.

    Parameters
    ----------
    price_df: pd.DataFrame
        Daily close price of the instrument being traded. This assumes equities only.
    fx_rate: pd.DataFrame or float
        Daily FX rate to GBP
    asset_class: str
        Asset class to calculated the block value for.
        Currently only equities are supported.

    Returns
    -------
    pd.Series
        Time series of the instrument's vol contribution per block.

    Raises
    ------
    NotImplementedError
         If using an asset_class that is not equity.
    """
    # TODO: this currently only works for equities
    if asset_class == 'equity':
        # Assuming we trade in blocks of 100 shares
        # This is the value of a 1% move in the price of the underlying
        block_value = price_df.copy()
    else:
        raise NotImplementedError(f"Block value calculations for {asset_class} "
                                  f"are not yet implemented.")

    # Sanitise inputs
    assert block_value.shape[1] == 1
    check_nonempty_dataframe(block_value, 'block_value')

    price_returns = returns(price_df, 'arithmetic', percent=True)
    price_vol = price_returns.ewm(span=VOL_SPAN).std()  # TODO: support buffering price_vol
    instrument_currency_vol = block_value * price_vol
    instrument_value_vol = convert_currency(instrument_currency_vol, fx_rate)
    instrument_value_vol.columns = ['instrument_value_volatility']
    instrument_value_vol = instrument_value_vol.iloc[:, 0]

    return instrument_value_vol


def get_cash_vol_target_daily(vol_target, trading_capital):
    """
    Convert the annualised percentage vol target to an daily cash amount.

    Parameters
    ----------
    vol_target: float
        The target annualised percentage volatility.
    trading_capital: pd.DataFrame or float
        Timeseries of capital available to invest. GBP amount.
        TODO: This is currently a scalar starting amount which does not reflect
         rolling up/ rolling down losses.
         The current implementation is equivalent to adding capital following losses and
         removing gains, so this does not reflect any compounding

    Returns
    -------
    cash_vol_target_daily: float, pd.Series
        The daily cash vol target.
        Returns a float if trading_capital is a scalar or a pd.Series if trading_capital
        is a timeseries
    """
    # Sanitise inputs
    check_scalar_is_above_min_threshold(vol_target, 'vol_target', 1)  # pct not decimal
    if isinstance(trading_capital, (int, float)):
        check_scalar_is_above_min_threshold(trading_capital, 'trading_capital', 0)
    else:
        check_nonempty_dataframe(trading_capital, 'trading_capital')

    cash_vol_target_annualised = (vol_target / 100.) * trading_capital
    cash_vol_target_daily = cash_vol_target_annualised / np.sqrt(TRADING_DAYS_PER_YEAR)

    if isinstance(cash_vol_target_daily, pd.DataFrame):
        assert cash_vol_target_daily.shape[1] == 1
        cash_vol_target_daily.columns = ['cash_vol_target_daily']
        cash_vol_target_daily = cash_vol_target_daily.iloc[:, 0]

    return cash_vol_target_daily
