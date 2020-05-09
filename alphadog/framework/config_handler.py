"""
Classes to store configurations and helper functions for Strategy and Instrument configs.
"""
import abc
import json
import os

from alphadog.constants import PROJECT_DIR
from alphadog.framework.signals_config import PARAMETERISED_STRATEGIES
from alphadog.framework.weights import STRATEGY_WEIGHTS, INSTRUMENT_WEIGHTS


class BaseConfiguration:
    """
    Generic class to hold user-defined hierarchies (Instruments, Strategies)
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.identifier} Configuration {self.__class__}"

    @classmethod
    def from_config(cls, input_dict):
        """
        Initialise a config object from an individual configuration dict.

        Parameters
        ----------
        input_dict: dict
            The config dictionary for an individual strategy/instrument.

        Returns
        -------
        :class:`BaseConfiguration`
            A BaseConfiguration object for the given configuration.

        Examples
        --------
        >>> from alphadog.framework.config_handler import PARAMETERISED_STRATEGIES, BaseConfiguration
        >>> vmom1_dict = PARAMETERISED_STRATEGIES["VMOM1"]
        >>> strat = BaseConfiguration.from_config(vmom1_dict)
        """
        return cls(**input_dict)

    @classmethod
    def from_identifier(cls, identifier):
        """
        Initialise a config object from an identifier.

        Looks up the given identifier in the reference_config for the class.

        Parameters
        ----------
        identifier: str
            The identifier to create a config object for.

        Returns
        -------
        :class:`BaseConfiguration`
            A BaseConfiguration object for the given identifier.

        Examples
        --------
        >>> from alphadog.framework.config_handler import Strategy
        >>> strat = Strategy.from_identifier('VMOM1')
        """
        input_dict = cls.reference_config()[identifier]
        return cls(**input_dict)

    @classmethod
    @abc.abstractmethod
    def reference_config(cls):
        """
        Returns the full config file.

        Abstract method to be implemented by the child class.
        """
        pass

    @property
    @abc.abstractmethod
    def identifier(self):
        """
        Returns the identifier of the strategy/instrument.

        Abstract method to be implemented by the child class.
        """
        pass

    @property
    @abc.abstractmethod
    def weight_config(self):
        """
        Returns the manually defined weights config, which currently give the level_1 weights.

        Abstract method to be implemented by the child class.
        """
        pass

    def depth(self):
        """
        The depth of the hierarchy where this object resides.

        Returns
        -------
        int
            The number of levels this object has in its hierarchy
        """
        object_dict = self.reference_config()[self.identifier]
        return hierarchy_depth(object_dict, self.identifier)

    def siblings(self):
        """
        The objects which are at the same level of the hierarchy as this one.

        Returns
        -------
        list
            The sibling objects of the given object
        """
        return get_siblings(self.reference_config(), self.identifier)


class Strategy(BaseConfiguration):
    """
    Class for configuration of a strategy.

    Holds all of the details of the Strategy, without reference to a specific
    instrument or dataset.

    Examples
    --------
    >>> from alphadog.framework.config_handler import PARAMETERISED_STRATEGIES, Strategy
    >>> vmom1_dict = PARAMETERISED_STRATEGIES["VMOM1"]
    >>> strat = Strategy.from_config(vmom1_dict)
    """
    def __init__(self, **kwargs):
        """
        signal_func: function
            Parameterised function to run the signal.
        raw_signal_func: function
            The raw function (without parameters) used to create the signal_func.
        required_data_fixtures: list(str)
            Names of data inputs required, e.g. price_df
        params: dict
            Kwargs which were passed pass to the function.
        strategy_name: str
            Name to identify this strategy.
        hierarchy_1: str
            Level 1 of the hierarchy
        hierarchy_2: str
            Level 2 of the hierarchy
        """
        super().__init__(**kwargs)

    @classmethod
    def reference_config(cls):
        """
        Returns the full strategy config file.
        """
        return PARAMETERISED_STRATEGIES

    @property
    def weight_config(self):
        """dict: The level_1 weights."""
        return STRATEGY_WEIGHTS

    @property
    def identifier(self):
        """
        Returns the identifier of the strategy.
        """
        return self.strategy_name


class Instrument(BaseConfiguration):
    """
    Class for configuration of an instrument.

    Holds all of the details of the Instrument, without reference to a specific
    strategy or dataset.

    Examples
    --------
    >>> from alphadog.framework.config_handler import load_default_instrument_config, Instrument
    >>> instrument_config = load_default_instrument_config()
    >>> ftse_dict = instrument_config["FTSE100"]
    >>> inst = Instrument.from_config(ftse_dict)
    """
    def __init__(self, **kwargs):
        """
        traded_instrument: str
            Long name of the instrument.
        traded_info_link: str
            URL of the traded instrument.
        cost: int
            Cost in bps.
        index: str
            The index that this instrument is tracking.
        instrument_id: str
            Name to identify this instrument.
        yfinance_symbol: str
            Yahoo finance symbol to identify this instrument.
        yfinance_link:
            Yahoo finance URL for this instrument.
        currency: str
            ISO currency code.
        is_traded: bool
            Flag whether to include this instrument in the portfolio.
        hierarchy_1: str
            Level 1 of the hierarchy
        hierarchy_2: str
            Level 2 of the hierarchy
        signals: list(str)
            List of signal names to include for this instrument.
            Signal names should correspond to signals in signals_config.
        """
        super().__init__(**kwargs)

    @classmethod
    def reference_config(cls):
        """
        Returns the full strategy config file.
        """
        return load_default_instrument_config()

    @property
    def weight_config(self):
        """dict: The level_1 weights."""
        return INSTRUMENT_WEIGHTS

    @property
    def identifier(self):
        """
        Returns the identifier of the instrument.
        """
        return self.instrument_id

    @property
    def strategies(self):
        """
        dict:
            The strategies to run on this instrument.
            This is in the format: {strategy_name: Strategy object}

            If the instrument is not traded, this returns an empty dict.
        """
        if self.is_traded:
            return {sig_name: Strategy.from_identifier(sig_name)
                    for sig_name in self.signals}
        else:
            return {}

    @property
    def required_data_fixtures(self):
        """
        List of unique data fixtures required for the signals run on this object.

        Returns
        -------
        list(str)
            List of data fixture names.
        """
        if not self.is_traded:
            return []

        req_fixtures = []
        for strat in self.strategies.values():
            tmp_fixtures = strat.required_data_fixtures
            if isinstance(tmp_fixtures, list):
                req_fixtures.extend(tmp_fixtures)
            elif isinstance(tmp_fixtures, str):
                req_fixtures.append(tmp_fixtures)

        return list(set(req_fixtures))


def hierarchy_depth(object_dict, name=None):
    """
    Return the number of levels an instrument has in its hierarchy.

    Parameters
    ----------
    object_dict: dict
        A single dict from the full config.
    name: str, optional
        Name of the object. Only used for logging errors.

    Returns
    -------
    depth: int
        The number of instruments in the hierarchy

    Examples
    --------
    >>> instrument_config = load_default_instrument_config()
    >>> instrument_dict = instrument_config['FTSE100']
    >>> hierarchy_depth(instrument_dict)
    """
    levels = [int(key.split('_')[-1]) for key in object_dict.keys() if 'hierarchy' in key]
    if not levels:
        return 0

    depth = max(levels)
    if depth != len(levels):
        raise ValueError(f"{name} has skipped a level in its hierarchy")

    return depth


def get_siblings(full_config, target_name):
    """
    Return a list of instruments at the same hierarchy level as this instrument.

    Parameters
    ----------
    full_config: dict, optional
        The nested dict specifying the full config.
    target_name: str
        A single target object to get siblings for.

    Returns
    -------
    list
        Objects which are at the same hierarchy level as the given instrument.
    """
    # FIXME: this loops through everything. We could track which siblings we have already
    #  found and skip these
    target_instrument = full_config[target_name]
    target_levels = [key for key in target_instrument.keys() if 'hierarchy' in key]
    target_values = [target_instrument[level] for level in target_levels]

    siblings = []
    for instrument_name, instrument_dict in full_config.items():
        instrument_levels = [key for key in instrument_dict.keys() if 'hierarchy' in key]
        instrument_values = [instrument_dict[level] for level in instrument_levels]

        if instrument_values == target_values:
            siblings.append(instrument_name)

    return siblings


def load_default_instrument_config():
    """
    Returns the default instrument config as specified in the json config file.

    Returns
    -------
    instrument_config: dict
        Specifies details of every instrument to run in the portfolio.
    """
    config_path = os.path.join(PROJECT_DIR, "alphadog/framework/instrument_config.json")

    with open(config_path) as config_file:
        instrument_config = json.load(config_file)

    return instrument_config
