import numpy as np
import pandas as pd
import pytest

from alphadog.framework.config_handler import Instrument, Strategy
from alphadog.framework.portfolio import (
    get_instrument_value_volatility, get_cash_vol_target_daily, get_vol_scalar,
    get_weights_from_config, get_diversification_multiplier, combine_signals,
    Forecast, Subsystem, Portfolio
)
from alphadog.internals.exceptions import InputDataError, DimensionMismatchError


def mock_signal(df, multiplier):
    return df * multiplier


@pytest.fixture
def expected_instrument_value_vol():
    df = pd.DataFrame(
        data=[np.nan, 28.99137802864845, 41.97760597014876, 55.435884849984355, 69.39249225768688,
              83.84757224998215, 98.79218919575786, 114.2132548210345, 130.09536197467733,
              146.42160539439644, 163.17401118675886, 180.33379609578282, 197.8815456574169,
              215.79735067437105, 234.06092074218242, 252.65168419493494, 271.5488793177745,
              290.7316393711832, 310.1790727500412, 329.8703389323459],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['instrument_value_volatility']
    )
    return df.iloc[:, 0]


class TestPortfolio:
    # TODO TEST
    def test_instantiation_with_config(self):
        """Test a Portfolio instantiates with an input config."""
        # TODO TEST
        # instument_config
        # vol_target
        pass

    def test_instantiation_default_config(self):
        """Test a Portfolio instantiates with a default config."""
        # TODO TEST
        pass

    def test_instantiation_with_input_vol_target(self):
        """Test a Portfolio instantiates with an input vol_target."""
        # TODO TEST
        pass

    def test_properties(self):
        """Test the properties of the Portfolio are present."""
        # TODO TEST
        # traded_instruments
        # instruments
        # instrument_weights
        pass

    def test_untraded_instrument(self):
        """Test an instrument with is_traded=False is not included in the Portfolio"""
        # TODO TEST
        pass

    def test_subsystem_lists(self):
        """Test the list of Subsystems in the Portfolio."""
        # TODO TEST
        # subsystems
        # pweights
        # diversification mults
        pass

    def test_combined_position(self):
        """Test the combined subsystems and final target position."""
        # TODO TEST
        # target_position
        pass


class TestSubsystem:

    def test_instantiation(self, mock_instrument):
        """Test a Subsystem instantiates with default vol_target."""
        actual = Subsystem(mock_instrument)
        assert actual.instrument_id == "FTSE100"
        assert actual.vol_target == 10

    @pytest.mark.parametrize('input_vol', [5, 10, 50])
    def test_instantiation_with_vol_target(self, mock_instrument, input_vol):
        """Test a Subsystem instantiates with input vol_target."""
        actual = Subsystem(mock_instrument, vol_target=input_vol)
        assert actual.instrument_id == "FTSE100"
        assert actual.vol_target == input_vol

    def test_properties(self, mock_instrument):
        """Test the properties of the Subsystem are present."""
        actual = Subsystem(mock_instrument)
        expected_strategies = ["VMOM1", "VMOM2", "VMOM3", "VMOM4", "VMOM5", "VMOM6",
                               "VBO1", "VBO2", "VBO3", "VBO4", "VBO5", "VBO6", "BLONG"]
        assert actual.currency == 'GBP'
        assert actual.instrument_id == 'FTSE100'
        assert actual.is_traded is True
        assert list(actual.strategies.keys()) == expected_strategies
        assert list(actual.strategy_weights.keys()) == expected_strategies
        assert sum(actual.strategy_weights.values()) == 1.
        assert actual.fx_rate == 1.
        assert actual.trading_capital == 10000.

    def test_data_fixtures(self, mock_instrument):
        """Test the data fixtures of the Subsystem."""
        actual = Subsystem(mock_instrument)
        assert actual.required_data_fixtures == ['price_df']
        assert list(actual.data.keys()) == ['price_df']
        assert not actual.data['price_df'].dropna().empty
        assert hasattr(actual, 'load_data_fixtures')

    def test_forecast_lists(self, mock_instrument):
        """Test the list of Forecasts per strategy in the Subsystem."""
        actual = Subsystem(mock_instrument)
        expected_num_signals = 13
        assert len(actual.forecast_list) == expected_num_signals
        assert len(actual.capped_forecasts) == expected_num_signals
        assert len(actual.fweights) == expected_num_signals

        assert all([not fc.raw_forecast.dropna().empty for fc in actual.forecast_list])
        assert all([not capped_fc.dropna().empty for capped_fc in actual.capped_forecasts])
        assert all([fw > 0 for fw in actual.fweights])
        assert sum(actual.fweights) == 1.

    def test_capped_forecasts(self, mock_instrument):
        """Test the capped forecasts have the expected shape and non-identical values."""
        actual = Subsystem(mock_instrument)
        capped_fc = pd.concat(actual.capped_forecasts, axis=1)
        expected_strategies = ["VMOM1", "VMOM2", "VMOM3", "VMOM4", "VMOM5", "VMOM6",
                               "VBO1", "VBO2", "VBO3", "VBO4", "VBO5", "VBO6", "BLONG"]
        expected_forecast_names = set([f"FTSE100|{strat}" for strat in expected_strategies])
        assert set(capped_fc.columns) == expected_forecast_names

        # Check there are no identical columns of values
        for col_number in range(capped_fc.shape[1]):
            test_array = capped_fc.iloc[:, col_number].values
            remaining_df = capped_fc.loc[:, capped_fc.columns[col_number+1:]]
            for remaining_col_number in range(remaining_df.shape[1]):
                assert not all(test_array == remaining_df.values[:, remaining_col_number])

    def test_position(self, mock_instrument):
        """Test the combined forecasts and final position."""
        actual = Subsystem(mock_instrument)

        assert set(actual.combined_forecast.columns) == {'combined'}
        assert not actual.combined_forecast.dropna().empty

        set(actual.vol_scalar.columns) == {'vol_scalar'}
        assert not actual.vol_scalar.dropna().empty

        assert actual.subsystem_position.shape[1] == 1
        set(actual.subsystem_position.columns) == {'FTSE100'}
        assert not actual.subsystem_position.dropna().empty


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


class TestCombineSignals:

    def test_success(self, mock_signal_list):
        weights = [0.3, 0.3, 0.1, 0.3]
        actual = combine_signals(mock_signal_list, weights)
        expected = pd.DataFrame(
            data=[8.403732233905812, 9.296163975559525, 4.164681461050669, 1.933602106916382,
                  4.387789396464097, 8.626840169319241, 9.519271910972956, 9.742379846386383,
                  6.618868750598383, 4.834005267290954, 3.7184655902238104, 3.7184655902238104,
                  5.50332907353124, 2.3798179777432398, 6.841976686011811, -2.5285566013521907,
                  -5.6520676971401915, -2.5285566013521907, 6.172652879771525, 6.172652879771526],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['combined']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_result_is_clipped(self, mock_signal_list):
        """Test that a results outside the allowed min or max forecast is clipped."""
        const_sig = mock_signal_list[2]
        signals = mock_signal_list[:3] + [const_sig] * 7
        weights = [1/10] * 10
        actual = combine_signals(signals, weights)
        expected = pd.DataFrame(
            data=[20.0, 20.0, 19.75, 18.5, 19.75, 20.0, 20.0, 20.0, 20.0, 20.0,
                  20.0, 20.0, 20.0, 20.0, 20.0, 19.75, 18.5, 19.75, 20.0, 20.0],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['combined']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_len_mismatch_raises(self, mock_signal_list):
        """The signals list and weights list must be the same length."""
        weights = [0.3, 0.7]
        expected_msg = "Number of weights does not equal number of signals." \
                       " Got 2 weights but 4 signals."
        with pytest.raises(DimensionMismatchError, match=expected_msg):
            combine_signals(mock_signal_list, weights)

    def test_weights_dont_sum_to_1_raises(self, mock_signal_list):
        weights = [0.9, 0.3, 1, 1.5]
        expected_msg = "Weights must sum to 1."
        with pytest.raises(InputDataError, match=expected_msg):
            combine_signals(mock_signal_list, weights)


class TestGetDiversificationMultiplier:

    def test_success(self, mock_signal_list):
        weights = [0.3, 0.3, 0.1, 0.3]
        actual = get_diversification_multiplier(mock_signal_list, weights)
        expected = 1.4873862360895242
        assert actual == expected

    def test_len_mismatch_raises(self, mock_signal_list):
        """The signals list and weights list must be the same length."""
        weights = [0.3, 0.7]
        expected_msg = "Number of weights does not equal number of signals." \
                       " Got 2 weights but 4 signals."
        with pytest.raises(DimensionMismatchError, match=expected_msg):
            get_diversification_multiplier(mock_signal_list, weights)

    def test_weights_dont_sum_to_1_raises(self, mock_signal_list):
        weights = [0.9, 0.3, 1, 1.5]
        expected_msg = "Weights must sum to 1."
        with pytest.raises(InputDataError, match=expected_msg):
            get_diversification_multiplier(mock_signal_list, weights)

    def test_both_lists_empty_raises(self):
        signals = []
        weights = []
        expected_msg = "Weights must sum to 1."
        with pytest.raises(InputDataError, match=expected_msg):
            get_diversification_multiplier(signals, weights)

    def test_multiplier_capped_above_max(self, mock_signal_list):
        """Test that large diversification multipliers are capped at the maximum allowed value."""
        const_sig = mock_signal_list[2]
        signals = mock_signal_list[:3] + [const_sig] * 27
        weights = [1/30] * 30
        actual = get_diversification_multiplier(signals, weights)
        expected = 2.5
        assert actual == expected


class TestGetWeightsFromConfig:

    def test_level_1(self):
        weights_config = {'bias': {'weight': 0.6}, 'trend': {'weight': 0.4}}
        strat_dict = {'strategy_name': 'BLONG', 'hierarchy_1': 'bias'}
        strat = Strategy.from_config(strat_dict)

        actual = get_weights_from_config(strat, weights_config)
        expected = 0.6 * 1.
        assert actual == expected

    def test_level_2(self, mock_instrument, mock_instrument_weights_config):
        actual = get_weights_from_config(mock_instrument, mock_instrument_weights_config)
        expected = 0.7 * 0.3 * 0.333333333333333333
        assert actual == expected

    def test_level_3(self, mock_instrument_dict, mock_instrument_weights_config):
        level3_dict = mock_instrument_dict.copy()
        level3_dict['hierarchy_3'] = 'large_cap'
        level3_inst = Instrument.from_config(level3_dict)
        level3_inst.depth = lambda: 3

        weights_config = mock_instrument_weights_config.copy()
        weights_config['equity']['uk']['large_cap'] = {'weight': 0.8}
        weights_config['equity']['uk']['small_cap'] = {'weight': 0.2}

        actual = get_weights_from_config(level3_inst, weights_config)
        expected = 0.7 * 0.3 * 0.8 * 0.333333333333333333
        assert actual == expected

    def test_higher_level_raises(self, mock_instrument_dict, mock_instrument_weights_config):
        level4_dict = mock_instrument_dict.copy()
        level4_dict['hierarchy_3'] = 'large_cap'
        level4_dict['hierarchy_4'] = 'TOOHIGH'
        level4_inst = Instrument.from_config(level4_dict)
        level4_inst.depth = lambda: 4

        weights_config = mock_instrument_weights_config.copy()
        weights_config['equity']['uk']['large_cap'] = {'weight': 0.8}
        weights_config['equity']['uk']['small_cap'] = {'weight': 0.2}

        with pytest.raises(NotImplementedError):
            get_weights_from_config(level4_inst, weights_config)


class TestGetVolScalar:

    def test_scalar_fxrate_and_trading_capital(self, mock_price):
        """Test success when fx_rate and trading_capital inputs are scalars."""
        fx_rate = 1.
        vol_target = 10.
        trading_capital = 10000.
        actual = get_vol_scalar(mock_price, fx_rate, vol_target, trading_capital)
        expected = pd.DataFrame(
            data=[np.nan, 2.172855625304263, 1.5006591581155853, 1.136341180542895,
                  0.9077938662434062, 0.7512928179609233, 0.6376423009520289, 0.5515478823677624,
                  0.4842146397742666, 0.4302239322208797, 0.3860546074507667, 0.34931931894458884,
                  0.31834236298078, 0.2919131242251744, 0.2691353970373339, 0.249331719420749,
                  0.2319806253413172, 0.21667431508699794, 0.2030893905135734,
                  0.19096618095084583],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['vol_scalar']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_df_fxrate_and_trading_capital(self, mock_price, mock_fx_rate, mock_trading_capital):
        """Test success when fx_rate and trading_capital inputs are DataFrames."""
        vol_target = 10.
        actual = get_vol_scalar(mock_price, mock_fx_rate, vol_target, mock_trading_capital)
        expected = pd.DataFrame(
            data=[np.nan, 2.992914084441134, 2.1795378022643632, 1.3024797026090238,
                  1.0446419634561637, 0.8133981598099412, 0.9793558954850053, 0.9144397998700308,
                  0.9333358515309687, 1.1078252406872144, np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['vol_scalar']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_df_fxrate_scalar_trading_capital(self, mock_price, mock_fx_rate):
        """Test success when fx_rate is a DataFrame and trading_capital is a scalar."""
        vol_target = 10.
        trading_capital = 10000
        actual = get_vol_scalar(mock_price, mock_fx_rate, vol_target, trading_capital)
        expected = pd.DataFrame(
            data=[np.nan, 2.720830985855576, 1.895250262838577, 1.447199669565582,
                  1.1607132927290706, 0.9569390115411073, 0.8161299129041711, 0.7034152306692544,
                  0.6222239010206458, 0.5539126203436072, 0.4969167298890033, 0.45015376152653197,
                  0.4133242832780837, 0.3752096712405841, 0.34695809853981424, 0.3230941031757794,
                  0.3034806715611162, 0.28431218355464893, 0.2682464542511866, 0.2513043570875718],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['vol_scalar']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_scalar_fxrate_df_trading_capital(self, mock_price, mock_trading_capital):
        """Test success when trading_capital is a DataFrame."""
        vol_target = 10.
        fx_rate = 1.
        actual = get_vol_scalar(mock_price, fx_rate, vol_target, mock_trading_capital)
        expected = pd.DataFrame(
            data=[np.nan, 2.3901411878346894, 1.7257580318329229, 1.0227070624886054,
                  0.8170144796190656, 0.6385988952667848, 0.7651707611424347, 0.7170122470780911,
                  0.7263219596613999, 0.8604478644417594, np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['vol_scalar']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_empty_price_df_raises(self, mock_price, mock_fx_rate, mock_trading_capital):
        """Test an error is raised when price is empty."""
        vol_target = 10.
        empty_price = mock_price.head(0)
        with pytest.raises(InputDataError):
            get_vol_scalar(empty_price, mock_fx_rate, vol_target, mock_trading_capital)

    def test_empty_fxrate_df_raises(self, mock_price, mock_fx_rate, mock_trading_capital):
        """Test an error is raised when fx_rate is empty."""
        vol_target = 10.
        empty_fx = mock_fx_rate.head(0)
        with pytest.raises(InputDataError):
            get_vol_scalar(mock_price, empty_fx, vol_target, mock_trading_capital)

    def test_negative_vol_target_raises(self, mock_price, mock_fx_rate, mock_trading_capital):
        vol_target = -69
        with pytest.raises(InputDataError):
            get_vol_scalar(mock_price, mock_fx_rate, vol_target, mock_trading_capital)

    def test_empty_trading_capital_df_raises(self, mock_price, mock_fx_rate, mock_trading_capital):
        """Test an error is raised when trading_capital is empty."""
        vol_target = 10.
        empty_trading_capital = mock_trading_capital.head(0)
        with pytest.raises(InputDataError):
            get_vol_scalar(mock_price, mock_fx_rate, vol_target, empty_trading_capital)


class TestGetInstrumentValueVolatility:

    @pytest.mark.parametrize('fx_rate', [1., 0.5, 2.])
    def test_scalar_fx_rate(self, mock_price, expected_instrument_value_vol, fx_rate):
        actual = get_instrument_value_volatility(mock_price, fx_rate)
        expected = expected_instrument_value_vol * fx_rate
        pd.testing.assert_series_equal(actual, expected)

    def test_dataframe_fx_rate(self, mock_price, mock_fx_rate, expected_instrument_value_vol):
        actual = get_instrument_value_volatility(mock_price, mock_fx_rate)
        expected = expected_instrument_value_vol * mock_fx_rate.values.flatten()
        pd.testing.assert_series_equal(actual, expected)

    @pytest.mark.parametrize(
        'indices, expected_values', [
            [range(10),
             [np.nan, 23.15251449367865, 33.23786840716379, 43.52825678420771, 54.27186819473691,
              65.82872897346098, 77.1863374186456, 89.55461310517315, 101.2402106886939,
              113.72566090982771, 126.7372544887556, 140.0652594275945, 153.69459651211568,
              167.60980226878397, 181.79511714045307, np.nan, np.nan, np.nan, np.nan, np.nan]],
            [range(10, 20), [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan, 126.76988929099296, 139.93902577032748,
                             152.40836646534248, 167.8903388246607, 181.56105621971088,
                             194.9713046932313, 207.57196335050682, 221.5665823647787,
                             234.8365759790562, 250.66847055468963]],
            [range(5, 17), [np.nan, np.nan, np.nan, np.nan, np.nan, 65.82872897346098, 77.1863374186456,
                            89.55461310517315, 101.2402106886939, 113.72566090982771,
                            126.76988929099296, 139.93902577032748, 152.40836646534248,
                            167.8903388246607, 181.56105621971088, 194.9713046932313,
                            207.57196335050682, 222.2352651353324, 237.10088321013149,
                            252.15288707988518]],
            [[4], [np.nan, np.nan, np.nan, np.nan, 54.27186819473691, 65.57718625671104,
                   77.26537117000223, 89.32618659553108, 101.74758260039515, 114.51633757895746,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
        ]
    )
    def test_dataframe_fx_rate_shorter_than_price_series(
            self, mock_price, mock_fx_rate, indices, expected_values
    ):
        """Test when the fx_rate series is shorter than the price_df time series."""
        fx_rate = mock_fx_rate.iloc[indices]
        actual = get_instrument_value_volatility(mock_price, fx_rate)
        expected = pd.DataFrame(
            data=expected_values,
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['instrument_value_volatility']
        )
        expected = expected.iloc[:, 0]
        pd.testing.assert_series_equal(actual, expected)

    @pytest.mark.parametrize(
        'indices, expected_timestamps, expected_values', [
            [range(10), pd.bdate_range('2019-01-01', periods=10),
             [np.nan, 23.15251449367865, 33.23786840716379, 43.52825678420771, 54.27186819473691,
              65.82872897346098, 77.1863374186456, 89.55461310517315, 101.2402106886939,
              113.72566090982771]],
            [range(10, 20), pd.bdate_range('2019-01-15', periods=10),
             [np.nan, 27.984457972238804, 40.02904547968681, 53.15914595126174, 66.06133720889856,
              79.0840985064915, 91.93343407802803, 105.56147926487856, 119.01503158249241,
              133.97307955944714]],
            [range(5, 17), pd.bdate_range('2019-01-08', periods=12),
             [np.nan, 25.41327629448828, 36.83305046514527, 48.15650856586919, 60.02182472933455,
              72.37908764556778, 84.99564520894141, 97.32524555073087, 111.7573365763298,
              125.1689576008038, 138.5135228761037, 151.3615805724412]],
            [[4], pd.bdate_range('2019-01-07', periods=1), [np.nan]]
        ]
    )
    def test_dataframe_fx_rate_longer_than_price_series(
            self, mock_price, mock_fx_rate,
            indices, expected_timestamps, expected_values
    ):
        """Test when the fx_rate series is longer than the price_df time series."""
        price_subset = mock_price.iloc[indices]
        actual = get_instrument_value_volatility(price_subset, mock_fx_rate)

        expected = pd.DataFrame(
            data=expected_values,
            index=pd.DatetimeIndex(expected_timestamps, name='timestamp'),
            columns=['instrument_value_volatility']
        )
        expected = expected.iloc[:, 0]
        pd.testing.assert_series_equal(actual, expected)

    def test_empty_price_raises(self, mock_price, mock_fx_rate):
        empty_price = mock_price.head(0)
        with pytest.raises(InputDataError):
            get_instrument_value_volatility(empty_price, mock_fx_rate)

    def test_empty_df_fxrate_raises(self, mock_price, mock_fx_rate):
        empty_fx = mock_fx_rate.head(0)
        with pytest.raises(InputDataError):
            get_instrument_value_volatility(mock_price, empty_fx)

    def test_unsupported_asset_class_raises(self, mock_price):
        asset_class = "SNAKEOIL"
        expected_msg = "Block value calculations for SNAKEOIL are not yet implemented."
        with pytest.raises(NotImplementedError, match=expected_msg):
            get_instrument_value_volatility(mock_price, 1., asset_class)


class TestGetCashVolTargetDaily:

    def test_scalar_trading_capital(self):
        vol_target = 10
        trading_capital = 10000
        actual = get_cash_vol_target_daily(vol_target, trading_capital)
        expected = 10000 * (10/100) / np.sqrt(252)
        assert actual == expected

    def test_dataframe_trading_capital(self, mock_trading_capital):
        vol_target = 20
        actual = get_cash_vol_target_daily(vol_target, mock_trading_capital)
        expected = 0.2 / np.sqrt(252) * mock_trading_capital
        expected.columns = ['cash_vol_target_daily']
        expected = expected.iloc[:, 0]
        pd.testing.assert_series_equal(actual, expected)

    def test_empty_df_trading_capital_raises(self, mock_trading_capital):
        vol_target = 20
        empty_trading_capital = mock_trading_capital.head(0)
        with pytest.raises(InputDataError):
            get_cash_vol_target_daily(vol_target, empty_trading_capital)

    def test_negative_trading_capital_raises(self):
        trading_capital = -10000
        expected_msg = "Input trading_capital is below the threshold value 0. Got -10000"
        with pytest.raises(InputDataError, match=expected_msg):
            get_cash_vol_target_daily(10, trading_capital)

    def test_negative_vol_target_raises(self):
        vol_target = -10
        expected_msg = "Input vol_target is below the threshold value 1. Got -10"
        with pytest.raises(InputDataError, match=expected_msg):
            get_cash_vol_target_daily(vol_target, 10000)

    def test_decimal_vol_target_raises(self):
        vol_target = 0.25
        expected_msg = "Input vol_target is below the threshold value 1. Got 0.25"
        with pytest.raises(InputDataError, match=expected_msg):
            get_cash_vol_target_daily(vol_target, 10000)
