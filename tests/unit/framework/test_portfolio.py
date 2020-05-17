import pandas as pd
import pytest

from alphadog.framework.portfolio import (
    get_vol_scalar, Forecast
)
from alphadog.internals.exceptions import InputDataError


def mock_signal(df, multiplier):
    return df * multiplier


def test_portfolio():
    # TODO TEST
    pass


def test_subsystem():
    # TODO TEST
    pass


class TestForecast:

    def test_instantiation_single_col_input(self, mock_price):
        params = {'df': mock_price, 'multiplier': 2.}
        actual = Forecast(mock_signal, params, 'TESTID', 'TESTFORECAST')
        assert actual.params == params
        assert actual.instrument_id == 'TESTID'
        assert actual.name == 'TESTFORECAST'

    def test_run_signals_single_col_input(self, mock_price):
        new_row = pd.DataFrame([200., -150.], columns=['price'],
                               index=pd.to_datetime(['2019-10-29', '2019-01-30']))
        price_df = pd.concat([mock_price, new_row])
        params = {'df': price_df, 'multiplier': 2.}
        actual = Forecast(mock_signal, params, 'TESTID', 'TESTFORECAST')

        expected_forecast_scalar = pd.Series({'price': 10 / 121.818182})
        expected_scaled_df = pd.DataFrame(
            [6.567164179104478, 6.731343283582089, 6.895522388059701, 7.059701492537314,
             7.223880597014926, 7.388059701492537, 7.552238805970149, 7.7164179104477615,
             7.880597014925373, 8.044776119402986, 8.208955223880597, 8.373134328358208,
             8.537313432835822, 8.701492537313433, 8.865671641791044, 9.029850746268657,
             9.194029850746269, 9.35820895522388, 9.522388059701493, 9.686567164179104,
             32.83582089552239, -24.62686567164179],
            index=pd.to_datetime(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                  '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                  '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                  '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                  '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28',
                                  '2019-10-29', '2019-01-30']),
            columns=['price']
        )
        expected_capped_df = pd.DataFrame(
            [6.567164179104478, 6.731343283582089, 6.895522388059701, 7.059701492537314,
             7.223880597014926, 7.388059701492537, 7.552238805970149, 7.7164179104477615,
             7.880597014925373, 8.044776119402986, 8.208955223880597, 8.373134328358208,
             8.537313432835822, 8.701492537313433, 8.865671641791044, 9.029850746268657,
             9.194029850746269, 9.35820895522388, 9.522388059701493, 9.686567164179104,
             20, -20],
            index=pd.to_datetime(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                  '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                  '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                  '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                  '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28',
                                  '2019-10-29', '2019-01-30']),
            columns=['TESTFORECAST']
        )

        # Test run_signals outputs
        pd.testing.assert_frame_equal(actual.raw_forecast, price_df * 2)
        pd.testing.assert_series_equal(actual.forecast_scalar, expected_forecast_scalar)
        pd.testing.assert_frame_equal(actual.scaled_forecast, expected_scaled_df)
        pd.testing.assert_frame_equal(actual.capped_forecast, expected_capped_df)

    def test_instantiation_multi_col_input(self, mock_ohlcv):
        params = {'df': mock_ohlcv, 'multiplier': 2.}
        actual = Forecast(mock_signal, params, 'TESTID', 'TESTFORECAST')
        assert actual.params == params
        assert actual.instrument_id == 'TESTID'
        assert actual.name == 'TESTFORECAST'

    def test_run_signals_multi_col_input(self, mock_ohlcv):
        new_row = pd.DataFrame([[200., 202., 199, 201, 20000], [-150., -148, -151, -149, -15000]],
                               columns=['open', 'high', 'low', 'close', 'volume'],
                               index=pd.to_datetime(['2019-10-29', '2019-01-30']))
        ohlcv_df = pd.concat([mock_ohlcv, new_row])
        params = {'df': ohlcv_df, 'multiplier': 2.}
        actual = Forecast(mock_signal, params, 'TESTID', 'TESTFORECAST')

        expected_forecast_scalar = pd.Series({
            'open': 10 / 121.818182,
            'high': 10 / 125.454545,
            'low': 10 / 120.,
            'close': 10 / 123.636363,
            'volume': 10 / 12181.818182
        })
        expected_scaled_df = pd.DataFrame({
            'open': [6.567164179104478, 6.731343283582089, 6.895522388059701, 7.059701492537314,
                     7.223880597014926, 7.388059701492537, 7.552238805970149, 7.7164179104477615,
                     7.880597014925373, 8.044776119402986, 8.208955223880597, 8.373134328358208,
                     8.537313432835822, 8.701492537313433, 8.865671641791044, 9.029850746268657,
                     9.194029850746269, 9.35820895522388, 9.522388059701493, 9.686567164179104,
                     32.83582089552239, -24.62686567164179],
            'high': [6.695652173913044, 6.855072463768116, 7.014492753623189, 7.173913043478262,
                     7.333333333333334, 7.492753623188406, 7.6521739130434785, 7.811594202898552,
                     7.971014492753624, 8.130434782608697, 8.28985507246377, 8.449275362318842,
                     8.608695652173914, 8.768115942028986, 8.927536231884059, 9.08695652173913,
                     9.246376811594203, 9.405797101449275, 9.56521739130435, 9.724637681159422,
                     32.20289855072464, -23.594202898550726],
            'low': [6.5, 6.666666666666666, 6.833333333333333, 7.0, 7.166666666666666,
                    7.333333333333333, 7.5, 7.666666666666666, 7.833333333333333, 8.0,
                    8.166666666666666, 8.333333333333332, 8.5, 8.666666666666666,
                    8.833333333333332, 9.0, 9.166666666666666, 9.333333333333332, 9.5,
                    9.666666666666666, 33.166666666666664, -25.166666666666664],
            'close': [6.632352941176471, 6.794117647058824, 6.955882352941177, 7.11764705882353,
                      7.279411764705883, 7.4411764705882355, 7.602941176470589, 7.764705882352942,
                      7.926470588235294, 8.088235294117647, 8.25, 8.411764705882353,
                      8.573529411764707, 8.73529411764706, 8.897058823529413, 9.058823529411764,
                      9.220588235294118, 9.382352941176471, 9.544117647058824, 9.705882352941178,
                      32.51470588235294, -24.10294117647059],
            'volume': [6.567164179104477, 6.731343283582089, 6.895522388059701, 7.059701492537313,
                       7.223880597014925, 7.388059701492537, 7.552238805970148, 7.716417910447761,
                       7.880597014925373, 8.044776119402984, 8.208955223880597, 8.373134328358208,
                       8.53731343283582, 8.701492537313433, 8.865671641791044, 9.029850746268655,
                       9.194029850746269, 9.35820895522388, 9.522388059701491, 9.686567164179104,
                       32.83582089552239, -24.626865671641788]
        },
            index=pd.to_datetime(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                  '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                  '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                  '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                  '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28',
                                  '2019-10-29', '2019-01-30'])
        )
        expected_capped_df = pd.DataFrame(
            [6.592466694659694, 6.755708668931558, 6.918950643203421, 7.082192617475283,
             7.245434591747147, 7.408676566019009, 7.5719185402908735, 7.735160514562736,
             7.8984024888345985, 8.061644463106463, 8.224886437378325, 8.38812841165019,
             8.551370385922052, 8.714612360193914, 8.877854334465779, 9.041096308737643,
             9.204338283009504, 9.367580257281368, 9.530822231553232, 9.694064205825093,
             20.0, -20.0],
            index=pd.to_datetime(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                  '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                  '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                  '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                  '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28',
                                  '2019-10-29', '2019-01-30']),
            columns=['TESTFORECAST']
        )

        # Test run_signals outputs
        pd.testing.assert_frame_equal(actual.raw_forecast, ohlcv_df * 2)
        pd.testing.assert_series_equal(actual.forecast_scalar, expected_forecast_scalar)
        pd.testing.assert_frame_equal(actual.scaled_forecast, expected_scaled_df)
        pd.testing.assert_frame_equal(actual.capped_forecast, expected_capped_df)

    def test_empty_forecast_raises(self, mock_price):
        price_df = mock_price.head(0)
        params = {'df': price_df, 'multiplier': 2.}
        expected_msg = "TESTFORECAST has an empty raw forecast. Check inputs."
        with pytest.raises(InputDataError, match=expected_msg):
            Forecast(mock_signal, params, 'TESTID', 'TESTFORECAST')


def test_get_diversification_multiplier():
    # TODO TEST
    pass


def test_combine_signals():
    # TODO TEST
    pass


def test_get_weights_from_config():
    # TODO TEST
    pass


class TestGetVolScalar:
    # TODO TEST
    def test_scalar_fxrate_and_trading_capital(self, mock_price):
        """Test success when fx_rate and trading_capital inputs are DataFrames."""
        # TODO TEST
        pass

    def test_df_fxrate_and_trading_capital(self):
        """Test success when fx_rate and trading_capital inputs are DataFrames."""
        # TODO TEST
        pass

    def test_df_fxrate_scalar_trading_capital(self):
        """Test success when fx_rate is a DataFrame."""
        # TODO TEST
        pass

    def test_scalar_fxrate_df_trading_capital(self):
        """Test success when trading_capital is a DataFrame."""
        # TODO TEST
        pass

    def test_empty_price_df(self):
        """Test an error is raised when price is empty."""
        # TODO TEST
        pass
