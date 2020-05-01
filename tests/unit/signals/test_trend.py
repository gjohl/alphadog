import numpy as np
import pandas as pd
import pytest

from alphadog.signals.trend import momentum_signal, breakout_signal
from alphadog.internals.exceptions import ParameterError


@pytest.fixture
def expected_momentum_result():
    return pd.DataFrame(
        {'price': [np.nan, 0.2357022603955226, 0.39887749788506854, 0.5095484986130991,
                   0.5803325379213695, 0.622939407912391, 0.6467732559957836, 0.6585909726500113,
                   0.6629562994523351, 0.6628557502902326, 0.6602123184370691, 0.6562508368209105,
                   0.6517414636979532, 0.6471581531994168, 0.6427814832471975, 0.638765848688758,
                   0.6351838450791873, 0.632055894843941, 0.6293701690397264, 0.6270960125626317]},
        index=pd.DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28'],
                               dtype='datetime64[ns]', name='timestamp', freq='B')
    )


@pytest.fixture
def expected_breakout_result():
    return pd.DataFrame(
        {'price': [np.nan] + [20.] * 19},
        index=pd.DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28'],
                               dtype='datetime64[ns]', name='timestamp', freq='B')
    )


class TestMomentumSignal:

    @pytest.mark.parametrize('fast, slow', [[2, 6]])
    def test_single_column_df(self, mock_price, fast, slow, expected_momentum_result):
        """Test a single column input returns the expected output."""
        actual = momentum_signal(mock_price, fast, slow)
        pd.testing.assert_frame_equal(actual, expected_momentum_result)

    @pytest.mark.parametrize('fast, slow', [[2, 6]])
    def test_multiple_column_df(self, mock_ohlcv, fast, slow, expected_momentum_result):
        """Test that multi-column inputs are calculated columnwise."""
        actual = momentum_signal(mock_ohlcv, fast, slow)
        df_list = [expected_momentum_result] * 5
        expected = pd.concat(df_list, axis=1)
        expected.columns = ['open', 'high', 'low', 'close', 'volume']
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize('fast, slow', [[2, 6]])
    def test_intermittent_nans(self, mock_ohlcv, fast, slow):
        """Test that nans are handled correctly."""
        df = mock_ohlcv.copy()
        df.iloc[10:15] = np.nan
        actual = momentum_signal(df, fast, slow)

        expected_col = pd.DataFrame({'price': [
            np.nan, 0.2357022603955226, 0.39887749788506854, 0.5095484986130991,
            0.5803325379213695, 0.622939407912391, 0.6467732559957836, 0.6585909726500113,
            0.6629562994523351, 0.6628557502902326, np.nan, np.nan,
            np.nan, np.nan, np.nan, 0.45006020896312,
            0.36647431110068285, 0.36578642627155605, 0.4046491217537645, 0.458527608467524]},
            index=pd.DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                    '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                    '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                    '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                    '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28'],
                                   dtype='datetime64[ns]', name='timestamp', freq='B'))
        expected_list = [expected_col] * 5
        expected = pd.concat(expected_list, axis=1)
        expected.columns = ['open', 'high', 'low', 'close', 'volume']

        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize('fast, slow', [[2, 6]])
    def test_empty_df(self, fast, slow):
        """Test that an empty input returns an empty output."""
        df = pd.DataFrame(data=[], index=[], columns=['price'])
        actual = momentum_signal(df, fast, slow)
        assert actual.empty
        assert actual.columns == ['price']

    def test_fast_higher_than_slow_raises(self, mock_price):
        """Test that inputting incompatible speeds raises."""
        fast = 18
        slow = 6
        expected_msg = "Span of fast must be less than slow. Got fast=18, slow=6"
        with pytest.raises(ParameterError, match=expected_msg):
            momentum_signal(mock_price, fast, slow)


class TestBreakoutSignal:

    def test_single_column_df(self, mock_price, expected_breakout_result):
        """Test a single column input returns the expected output."""
        actual = breakout_signal(mock_price, lookback_period=4)
        pd.testing.assert_frame_equal(actual, expected_breakout_result)

    def test_multiple_column_df(self, mock_ohlcv, expected_breakout_result):
        """Test that multi-column inputs are calculated columnwise."""
        actual = breakout_signal(mock_ohlcv, lookback_period=4)
        df_list = [expected_breakout_result] * 5
        expected = pd.concat(df_list, axis=1)
        expected.columns = ['open', 'high', 'low', 'close', 'volume']
        pd.testing.assert_frame_equal(actual, expected)

    def test_intermittent_nans(self, mock_ohlcv):
        """Test that nans are handled correctly."""
        df = mock_ohlcv.copy()
        df.iloc[10:15] = np.nan
        actual = breakout_signal(df, lookback_period=4)

        expected_col = pd.DataFrame(
            {'price': [np.nan] + [20.] * 9 + [np.nan] * 5 + [20.] * 5},
            index=pd.DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                    '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                    '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                    '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                    '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28'],
                                   dtype='datetime64[ns]', name='timestamp', freq='B'))
        expected_list = [expected_col] * 5
        expected = pd.concat(expected_list, axis=1)
        expected.columns = ['open', 'high', 'low', 'close', 'volume']

        pd.testing.assert_frame_equal(actual, expected)

    def test_empty_df(self):
        """Test that an empty input returns an empty output."""
        df = pd.DataFrame(data=[], index=[], columns=['price'])
        actual = breakout_signal(df, lookback_period=80)
        assert actual.empty
        assert actual.columns == ['price']

    def test_smooth_longer_than_lookback_raises(self, mock_price):
        """Test that inconsistent smooth and ookback periods raises."""
        lookback_period = 10
        smooth_period = 20
        expected_msg = "Lookback period 10 must be greater than smooth period 20"
        with pytest.raises(ParameterError, match=expected_msg):
            breakout_signal(mock_price, lookback_period, smooth_period)
