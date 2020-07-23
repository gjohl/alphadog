import numpy as np
import pandas as pd

from alphadog.analytics.aggregation import cross_sectional_mean


class TestCrossSectionalMean:

    def test_single_column(self, mock_price):
        actual = cross_sectional_mean(mock_price, 'NEWNAME')
        expected = mock_price.copy()
        expected.columns = ['NEWNAME']
        pd.testing.assert_frame_equal(actual, expected)

    def test_multi_column(self, mock_ohlcv):
        input_df = mock_ohlcv[['open', 'high', 'low', 'close']]
        actual = cross_sectional_mean(input_df, 'COMBINED_DF')
        expected = pd.DataFrame(
            data=[40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5,
                  53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['COMBINED_DF']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_multicolumn_with_intermittent_nans(self, mock_ohlcv):
        input_df = mock_ohlcv[['open', 'high', 'low', 'close']]
        input_df.iloc[[13], 3] = np.nan
        input_df.iloc[[4, 6, 8, 9], 1] = np.nan
        input_df.iloc[[4, 9], 3] = np.nan
        input_df.iloc[15, :] = np.nan
        actual = cross_sectional_mean(input_df, 'COMBINED_DF')
        expected = pd.DataFrame(
            data=[40.5, 41.5, 42.5, 43.5, 43.5, 45.5, 46.0, 47.5, 48.0, 48.5, 50.5, 51.5, 52.5,
                  53.333333333333336, 54.5, np.nan, 56.5, 57.5, 58.5, 59.5],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['COMBINED_DF']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_empty_multicolumn_dataframe(self, mock_ohlcv):
        empty_df = mock_ohlcv.head(0)
        actual = cross_sectional_mean(empty_df, 'EMPTYTEST')
        assert actual.empty
        assert actual.columns == ['EMPTYTEST']

    def test_series(self, mock_price):
        price_series = mock_price.iloc[:, 0]
        actual = cross_sectional_mean(price_series, 'TESTSERIES')
        expected = mock_price.copy()
        expected.columns = ['TESTSERIES']
        pd.testing.assert_frame_equal(actual, expected)

    def test_empty_series(self, mock_price):
        empty_series = mock_price.iloc[:, 0].head(0)
        actual = cross_sectional_mean(empty_series, 'EMPTYSERIES')
        assert actual.empty
        assert actual.columns == ['EMPTYSERIES']
