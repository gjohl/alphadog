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


@pytest.fixture
def mock_fx_rate():
    return pd.DataFrame(
        data=[0.784, 0.7986, 0.7918, 0.7852, 0.7821, 0.7851, 0.7813, 0.7841, 0.7782, 0.7767,
              0.7769, 0.776, 0.7702, 0.778, 0.7757, 0.7717, 0.7644, 0.7621, 0.7571, 0.7599],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['MOCKFXRATE']
    )


@pytest.fixture
def mock_trading_capital():
    return pd.DataFrame(
        data=[10000, 11000, 11500, 9000, 9000, 8500, 12000, 13000, 15000, 20000],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=10), name='timestamp'),
        columns=['trading_capital']
    )


@pytest.fixture
def mock_forecast_signal():
    return pd.DataFrame(
        data=[10, 12, 14, 14, 15, 16, 20, 15, 10, 5,
              0, -3.5, -4, -12, -8, -16, -20, -15, -0.5, 0.5],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['combined']
    )


@pytest.fixture
def mock_signal_list():
    sig1 = pd.DataFrame(
        data=[10, 12, 14, 14, 15, 16, 20, 15, 10, 5,
              0, -3.5, -4, -12, -8, -16, -20, -15, -0.5, 0.5],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['SIG1']
    )
    sig2 = pd.DataFrame(
        data=[0.5, -0.5, -15, -20, -16, -8, -12, -4, -3.5,
              0, 5, 10, 15, 20, 16, 15, 14, 14, 12, 10],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['SIG2']
    )
    sig3 = pd.DataFrame(
        data=[10] * 20,
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['SIG3']
    )
    sig4 = pd.DataFrame(
        data=[5, 6, 7, 7, 7.5, 8, 10, 7.5, 5, 2.5,
              0, -1.5, -2, -6, 4, -8, -10, -8, -1, 0],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['SIG4']
    )
    return [sig1, sig2, sig3, sig4]

