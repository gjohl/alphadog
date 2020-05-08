from alphadog.data.yfinance_data import load_yfinance_data


def test_load_yfinance_data():
    instrument_id = 'FTSE100'
    actual_df = load_yfinance_data(instrument_id)
    expected_cols = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
    assert all(actual_df.columns == expected_cols)
    assert actual_df.index.name == 'timestamp'
    assert actual_df.index.dtype == 'datetime64[ns]'
