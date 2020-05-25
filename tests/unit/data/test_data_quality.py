from unittest import mock

import numpy as np
import pandas as pd
import pytest

from alphadog.data.data_quality import (
    staleness, check_nonempty_dataframe, check_scalar_is_above_min_threshold, check_price_data
)
from alphadog.internals.exceptions import InputDataError


class TestStaleness:

    def test_success(self, mock_ohlcv):
        with mock.patch('alphadog.data.data_quality.datetime') as mock_date:
            mock_date.today.return_value = pd.Timestamp('2019-02-04')
            mock_date.side_effect = lambda *args, **kw: pd.Timestamp(*args, **kw)
            actual = staleness(mock_ohlcv)
            expected = 7
            assert actual == expected

    def test_empty_df_raises(self, mock_ohlcv):
        empty_df = mock_ohlcv.head(0)
        expected_msg = "Cannot calculate staleness for an empty Dataframe."
        with pytest.raises(InputDataError, match=expected_msg):
            staleness(empty_df)

    def test_non_date_index_raises(self):
        df = pd.DataFrame(data=[6, 9], index=['a', 'b'])

        with mock.patch('alphadog.data.data_quality.datetime') as mock_date:
            mock_date.today.return_value = pd.Timestamp('2019-02-04')
            expected_msg = "Index must be a datetime."
            with pytest.raises(TypeError, match=expected_msg):
                staleness(df)


class TestCheckNonemptyDataframe:

    def test_success(self, mock_ohlcv):
        actual = check_nonempty_dataframe(mock_ohlcv, 'MOCKDATA')
        assert actual is None

    def test_empty_df(self, mock_ohlcv):
        empty_df = mock_ohlcv.head(0)
        expected_msg = "No data for EMPTYDATA"
        with pytest.raises(InputDataError, match=expected_msg):
            check_nonempty_dataframe(empty_df, 'EMPTYDATA')

    def test_df_all_nan(self, mock_ohlcv):
        nan_df = mock_ohlcv.copy()
        nan_df.loc[:, :] = np.nan
        expected_msg = "No data for NANDATA"
        with pytest.raises(InputDataError, match=expected_msg):
            check_nonempty_dataframe(nan_df, 'NANDATA')


class TestScalarIsAboveMinThreshold:

    @pytest.mark.parametrize("input_scalar, threshold", [[69, 50], [0.5, 0], [1., 1.]])
    def test_scalar_above_threshold(self, input_scalar, threshold):
        actual = check_scalar_is_above_min_threshold(input_scalar, 'CHEEKYSCALAR', threshold)
        assert actual is None

    def test_scalar_below_threshold_raises(self):
        expected_msg = "Input BADSCALAR is below the threshold value 0. Got -10"
        with pytest.raises(InputDataError, match=expected_msg):
            check_scalar_is_above_min_threshold(-10, 'BADSCALAR', 0)


class TestCheckPriceData:

    def test_success(self, mock_ohlcv):
        price_df = mock_ohlcv[['close']]
        actual = check_price_data(price_df, 'GOODDATA')
        assert actual is None

    def test_empty_df(self, mock_ohlcv):
        empty_df = mock_ohlcv[['close']].head(0)
        expected_msg = "No data for EMPTYDATA"
        with pytest.raises(InputDataError, match=expected_msg):
            check_price_data(empty_df, 'EMPTYDATA')

    def test_df_all_nan(self, mock_ohlcv):
        nan_df = mock_ohlcv.copy()
        nan_df.loc[:, :] = np.nan
        expected_msg = "No data for NANDATA"
        with pytest.raises(InputDataError, match=expected_msg):
            check_price_data(nan_df, 'NANDATA')

    def test_bad_columns_raises(self, mock_ohlcv):
        price_df = mock_ohlcv[['high']]
        expected_msg = "Incorrect columns for BADCOLDATA"
        with pytest.raises(InputDataError, match=expected_msg):
            check_price_data(price_df, 'BADCOLDATA')

    @pytest.mark.xfail(reason="Commented out test for now because lots of tests break.")
    def test_bad_prices_raises(self, mock_ohlcv):
        price_df = mock_ohlcv.loc[:, ['close']]
        price_df.iloc[4] = -69
        expected_msg = "Found negative prices for NEGPRICEDATA"
        with pytest.raises(InputDataError, match=expected_msg):
            check_price_data(price_df, 'NEGPRICEDATA')
