"""
Functionality to handle FINRA short volume data from http://regsho.finra.org/regsho-Index.html.
"""
from datetime import datetime

import requests
import pandas as pd

from alphadog.data.constants import (
    DATA_DIR, FINRA_DIR, FINRA_BASE_URL, FINRA_EXCHANGES
)


def get_finra_data(date, exchange):
    """Scrape FINRA short volume data.

    Retrieve the txt file for a given FINRA exchange-date and save it to disk.

    Parameters
    ----------
    date: str
        Date string in the format YYYYMMDD
    exchange: str
        Exchange code.
        Can be one of: ['CNMS', 'FNQC', 'FNRA', 'FNSQ', 'FNYX', 'FORF']

    Returns
    -------
    Write the file locally.
    """
    filename = f"{exchange}shvol{date}.txt"
    filepath = f"{DATA_DIR}{FINRA_DIR}{exchange}/{filename}"
    url = f"{FINRA_BASE_URL}{filename}"
    r = requests.get(url)

    if r.ok:
        txt_file = r.text
        with open(filepath, 'w') as f:
            f.write(txt_file)


def backfill_finra_data(start_date, end_date=None, exchanges=FINRA_EXCHANGES):
    """Backfill FINRA data for the given date range

    Parameters
    ----------
    start_date: date-like
        In a format that can be handled by pandas.
    end_date: date-like. Default None.
        In a format that can be handled by pandas.
        If None, use today's date.
    exchanges: list(str)
        Exchange codes.
        Can be: ['CNMS', 'FNQC', 'FNRA', 'FNSQ', 'FNYX', 'FORF']

    Returns
    -------
    Write the files locally.
    """
    if end_date is None:
        end_date = datetime.today().date()

    dates = pd.bdate_range(start=start_date, end=end_date)
    dates_str = [dt.strftime('%Y%m%d') for dt in dates]

    for exchange in exchanges:
        for date in dates_str:
            get_finra_data(date, exchange)


def load_finra_data(date, exchange='CNMS'):
    """Load a saved FINRA file for a given date and exchange.

    Parameters
    ----------
    date: str
        Date string in the format YYYYMMDD
    exchange: str
        Exchange code.
        Can be one of: ['CNMS', 'FNQC', 'FNRA', 'FNSQ', 'FNYX', 'FORF']

    Returns
    -------
    df: pd.DataFrame
        DataFrame of FINRA short volume data with columns:
        ['Date', 'Symbol', 'ShortVolume', 'ShortExemptVolume', 'TotalVolume', 'Market']
    """
    filename = f"{exchange}shvol{date}.txt"
    filepath = f"{DATA_DIR}{FINRA_DIR}{exchange}/{filename}"

    df = pd.read_csv(filepath, sep='|')
    df = df.dropna(subset=['Symbol'])
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format="%Y%m%d").dt.date

    return df
