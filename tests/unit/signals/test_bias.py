import pandas as pd

from alphadog.signals.bias import long_bias_signal, short_bias_signal


class TestLongBias:

    def test_single_column_df(self, mock_price):
        """Test a single column input returns the expected output."""
        actual = long_bias_signal(mock_price)

        index = pd.bdate_range('2019-01-01', periods=20, name='timestamp')
        expected = pd.DataFrame(data=[10] * 20, columns=['price'], index=index, dtype=float)

        pd.testing.assert_frame_equal(actual, expected)

    def test_multiple_column_df(self, mock_ohlcv):
        """Test that multi-column inputs are calculated columnwise."""
        actual = long_bias_signal(mock_ohlcv)

        vals = [10] * 20
        index = pd.bdate_range('2019-01-01', periods=20, name='timestamp')
        expected = pd.DataFrame({
            'open': vals,
            'high': vals,
            'low': vals,
            'close': vals,
            'volume': vals},
            index=index,
            dtype=float)

        pd.testing.assert_frame_equal(actual, expected)


class TestShortBias:

    def test_single_column_df(self, mock_price):
        """Test a single column input returns the expected output."""
        actual = short_bias_signal(mock_price)

        index = pd.bdate_range('2019-01-01', periods=20, name='timestamp')
        expected = pd.DataFrame(data=[-10] * 20, columns=['price'], index=index, dtype=float)

        pd.testing.assert_frame_equal(actual, expected)

    def test_multiple_column_df(self, mock_ohlcv):
        """Test that multi-column inputs are calculated columnwise."""
        actual = short_bias_signal(mock_ohlcv)

        vals = [-10] * 20
        index = pd.bdate_range('2019-01-01', periods=20, name='timestamp')
        expected = pd.DataFrame({
            'open': vals,
            'high': vals,
            'low': vals,
            'close': vals,
            'volume': vals},
            index=index,
            dtype=float)

        pd.testing.assert_frame_equal(actual, expected)
