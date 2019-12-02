"""
Mock data to use for unit tests.
"""
import pandas as pd
import pytest


@pytest.fixture
def mock_price():
    """
    Mock a DataFrame with a single column of 'price'.
    """
    data = [k for k in range(40, 60)] + [k for k in range(55, 45)]
    index = pd.bdate_range('2019-01-01', periods=len(data))
    index.name = 'timestamp'
    price_df = pd.DataFrame(data=data, columns=['price'], index=index, dtype=float)
    return price_df


@pytest.fixture
def mock_ohlcv():
    """
    Mock a DataFrame with columns: 'open', high, 'low', 'close', 'volume'.
    """
    data = [k for k in range(40, 60)] + [k for k in range(55, 45)]
    open_series = data.copy()
    high_series = [k + 2 for k in data]
    low_series = [k - 1 for k in data]
    close_series = [k + 1 for k in data]
    volume_series = [k * 100 for k in data]

    index = pd.bdate_range('2019-01-01', periods=len(data))
    index.name = 'timestamp'
    ohlcv_df = pd.DataFrame({
        'open': open_series,
        'high': high_series,
        'low': low_series,
        'close': close_series,
        'volume': volume_series},
        index=index,
        dtype=float)
    return ohlcv_df
