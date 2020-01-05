"""
Functionality to handle yahoo finance data.
"""
from datetime import datetime
import logging

import pandas as pd
import yfinance as yf

from .constants import YFINANCE_REQUIRED_COLS, YFINANCE_SYMBOL_MAPPING, DATA_DIR


def get_yfinance_data(symbol, **kwargs):
    """
    Retrieve data for the given symbol from yahoo finance and reshape.

    Parameters
    ----------
    symbol: str
        Yahoo finance symbol for the desired security
    kwargs: dict
        Optional arguments passed to the history function of the ticker.
        Typically set period='max' for backfills or set start and end for incremental updates.

    Returns
    -------
    df: pd.DataFrame
        DataFrame containing data for the given symbol and kwargs.

    Raises
    ------
    AssertionError
        The DataFrame must contain OHLCV columns
    """
    # Retrieve data
    ticker = yf.Ticker(symbol)
    df = ticker.history(**kwargs)
    df.columns = [k.lower().replace(' ', '_') for k in df.columns]
    df.index.name = 'timestamp'

    # Check expected columns
    for col in YFINANCE_REQUIRED_COLS:
        assert col in df.columns

    return df


def backfill_yfinance_data(symbol):
    """
    Retrieve the full backfill history available and save as CSV files data/csv_files

    Symbols should be mapped to an internal symbol in data/constants

    Parameters
    ----------
    symbol: str
        Yahoo finance symbol for the desired security

    Returns
    -------
    None
        Saves CSV files of data

    Raises
    ------
    ValueError
        If the given yfinance symbol is not mapped to an internal symbol in constants.
    """
    logging.info(f"Backfilling symbol {symbol}")

    if symbol in YFINANCE_SYMBOL_MAPPING:
        filename = YFINANCE_SYMBOL_MAPPING[symbol]
    else:
        raise ValueError(f"Unknown symbol. {symbol} is not mapped.")

    file_dir = f"{DATA_DIR}{filename}.csv"
    df = get_yfinance_data(symbol, period='max')
    df.to_csv(file_dir)

    logging.info(f"Successfully wrote {symbol} data to {file_dir}")


def update_yfinance_data(symbol):
    """
    Updates the csv file for the given symbol with any new data.

    Symbols should be mapped to an internal symbol in data/constants

    Parameters
    ----------
    symbol: str
        Yahoo finance symbol for the desired security

    Returns
    -------
    None
        Saves CSV files of data

    Raises
    ------
    ValueError
        If the given yfinance symbol is not mapped to an internal symbol in constants.
    """
    logging.info(f"Backfilling symbol {symbol}")

    if symbol in YFINANCE_SYMBOL_MAPPING:
        filename = YFINANCE_SYMBOL_MAPPING[symbol]
    else:
        raise ValueError(f"Unknown symbol. {symbol} is not mapped.")
    file_dir = f"{DATA_DIR}{filename}.csv"

    # Check the existing file
    old_df = pd.read_csv(file_dir)
    latest_date = old_df.index[-1].date()
    today = datetime.today().date()
    if latest_date == today:
        return

    start_date = latest_date + pd.Timedelta(1, 'day')
    new_df = get_yfinance_data(symbol, start=start_date.strftime("%Y-%m-%d"))

    if new_df.index[0].date() < start_date:
        # Using a recent date that there is no data for returns the most recent day of data
        return

    # Append the new data to the existing data
    df = pd.concat([old_df, new_df])
    df = df.drop_duplicates()
    df.to_csv(file_dir)

    logging.info(f"Successfully wrote {symbol} data to {file_dir}")


def load_yfinance_data(internal_symbol):
    """
    Load the saved yfiance data from the CSV file.

    Parameters
    ----------
    internal_symbol: str
        internal symbol for the desired security

    Returns
    -------
    df: pd.DataFrame
        DataFrame of data for the requested security.
    """
    file_dir = f"{DATA_DIR}{internal_symbol}.csv"
    df = pd.read_csv(file_dir, index_col='timestamp')
    return df
