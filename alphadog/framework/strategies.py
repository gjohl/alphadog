"""
Raw functionality which make up strategies once parameterised.



-- Momentum - VMOM
-- Breakout - VBO
-- Bias - BLONG, BSHORT
-- carry
-- value
-- fundamental
"""
from alphadog.framework.portfolio import hierarchy_depth, get_siblings
from alphadog.framework.signals_config import PARAMETERISED_STRATEGIES

# TODO: have the same thing for Instrument? Both could inherit from a generic Base class
#  and just change the init and reference_config


class Strategy:
    """
    Generic class for strategies.

    Holds all of the details of the Strategy, without reference to a specific instrument
    or dataset.
    """
    def __init__(self,
                 signal_func,
                 required_data_fixtures,
                 params={}, strategy_name='', raw_signal_func=None,
                 hierarchy_1=None, hierarchy_2=None):
        """
        signal_func: function
            Parameterised function to run the signal.
        required_data_fixtures: list(str)
            Names of data inputs required, e.g. price_df
        params: dict
            Kwargs which were passed pass to the function.
        strategy_name: str
            Name to identify this strategy.
        """
        self.signal_func = signal_func
        self.raw_signal_func = raw_signal_func or signal_func
        self.required_data_fixtures = required_data_fixtures
        self.params = params
        self.strategy_name = strategy_name
        self.hierarchy_1 = hierarchy_1
        self.hierarchy_2 = hierarchy_2

    @classmethod
    def from_config(cls, strategy_dict):
        """
        Initialise a Strategy object from an individual strategy configuration dict.

        Parameters
        ----------
        strategy_dict: dict
            The config dictionary for an individual strategy from signals_config.

        Returns
        -------
        :class:`Strategy`
            A Strategy object for the given configuration.

        Examples
        --------
        >>> from alphadog.framework.strategies import PARAMETERISED_STRATEGIES, Strategy
        >>> vmom1_dict = PARAMETERISED_STRATEGIES["VMOM1"]
        >>> strat = Strategy.from_config(vmom1_dict)
        """
        return cls(**strategy_dict)

    def depth(self):
        """
        The depth of the hierarchy where this Strategy resides.

        Returns
        -------
        int
            The number of levels this Strategy has in its hierarchy
        """
        name = self.strategy_name
        strategy_dict = PARAMETERISED_STRATEGIES[name]
        return hierarchy_depth(strategy_dict)

    def siblings(self):
        """
        The strategies which are at the same level of the hierarchy as this Strategy.

        Returns
        -------
        list
        """
        return get_siblings(PARAMETERISED_STRATEGIES, self.strategy_name)
