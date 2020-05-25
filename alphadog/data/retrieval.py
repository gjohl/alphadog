"""
Functionality to get normalised data.
"""
from alphadog.data.constants import PRICE_COLS
from alphadog.data.data_quality import (
    staleness, check_nonempty_dataframe, check_price_data
)
from alphadog.data.yfinance_data import load_yfinance_data


class BaseData:
    """
    Generic class to hold any data.
    """

    def __init__(self, input_df, name=None):
        """
        Create a data object from an input and run checks.

        Parameters
        ----------
        input_df: pd.DataFrame
           Input data.
        name: str, optional
           Name to assign to this data object
        """
        self.df = input_df
        self.name = name
        check_nonempty_dataframe(self.df, self.name)
        self.staleness = staleness(input_df)

    def __repr__(self):
        return f"{self.name} {self.__class__}"

    @classmethod
    def from_dataframe(cls, input_df, name=None):
        """
        Instantiate a BaseData object given a DataFrame.

        Parameters
        ----------
        input_df: pd.DataFrame
           Input data.
        name: str, optional
           Name to assign to this data object

        Returns
        -------close
        :class:`BaseData`
            A BaseData object for the given input DataFrame.
        """
        return cls(input_df, name)


class PriceData(BaseData):
    """
    Class for price data.
    """
    def __init__(self, input_df, name=None):
        price_df = input_df[PRICE_COLS]
        super().__init__(price_df, name)
        check_price_data(self.df, self.name)

    @classmethod
    def from_instrument_id(cls, instrument_id):
        """
        Instantiate a PriceData object given an instrument_id.

        This method loads the relevant instrument from the database.

        Parameters
        ----------
        instrument_id: str
            Instrument identifier to load.

        Returns
        -------
        :class:`PriceData`
            A PriceData object for the given instrument identifier.
        """
        input_df = load_yfinance_data(instrument_id)
        return cls(input_df, instrument_id)
