from alphadog.data.finra_data import load_finra_data


def test_load_finra_data():
    test_date = '20200102'
    actual_df = load_finra_data(test_date)
    expected_cols = ['Date', 'Symbol', 'ShortVolume', 'ShortExemptVolume', 'TotalVolume', 'Market']
    assert all(actual_df.columns == expected_cols)
    assert actual_df.shape[0] > 0
