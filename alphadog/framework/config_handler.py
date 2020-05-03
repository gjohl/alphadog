"""
Classes to store configurations and helper functions for Strategy and Instrument configs.
"""
import abc
import json
import os

from alphadog.constants import PROJECT_DIR
from alphadog.framework.signals_config import PARAMETERISED_STRATEGIES


class BaseConfiguration:
    """
    Generic class to hold user-defined hierarchies (Instruments, Strategies)
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

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

    @abc.abstractmethod
    def reference_config(self):
        """
        Returns the full config file.

        Abstract method to be implemented by the child class.
        """
        pass

    @abc.abstractmethod
    def identifier(self):
        """
        Returns the identifier of the strategy/instrument.

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
        object_dict = self.reference_config()[self.identifier()]
        return hierarchy_depth(object_dict, self.identifier())

    def siblings(self):
        """
        The objects which are at the same level of the hierarchy as this one.

        Returns
        -------
        list
            The sibling objects of the given object
        """
        return get_siblings(self.reference_config(), self.identifier())


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

    def reference_config(self):
        """
        Returns the full strategy config file.
        """
        return PARAMETERISED_STRATEGIES

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
        traded_instrument
        traded_info_link
        cost
        index

        instrument_id: str
            Name to identify this instrument.

        yfinance_symbol
        yfinance_link
        currency
        is_traded

        hierarchy_1: str
            Level 1 of the hierarchy
        hierarchy_2: str
            Level 2 of the hierarchy

        signals: list(str)
        """
        super().__init__(**kwargs)

    def reference_config(self):
        """
        Returns the full strategy config file.
        """
        return load_default_instrument_config()

    def identifier(self):
        """
        Returns the identifier of the strategy.
        """
        return self.instrument_id


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
