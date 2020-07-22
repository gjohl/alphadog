from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from alphadog.framework.config_handler import Instrument, Strategy
from alphadog.framework.constants import VOL_TARGET, STARTING_CAPITAL
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
        data=[np.nan, np.nan, 594.929754203148, 474.81957505440056, 373.96704418048523,
              353.6770705532807, 325.59866930537896, 287.70218188583124, 280.60988088458333,
              267.2045174011252, 391.4181926855501, 362.20789454712127, 335.9750188322304,
              350.69336349984775, 328.4574220665565, 308.1017924473117, 316.2940839435109,
              307.5503743456885, 289.8648586980328, 273.20222181341376],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['instrument_value_volatility']
    )
    return df.iloc[:, 0]


class TestPortfolio:

    def test_instantiation_with_config(self, mock_instrument_config):
        """Test a Portfolio instantiates with an input config."""
        actual = Portfolio(mock_instrument_config)
        assert actual.instrument_config == mock_instrument_config
        assert actual.vol_target == VOL_TARGET

    def test_instantiation_default_config(self):
        """Test a Portfolio instantiates with a default config."""
        actual = Portfolio()
        assert 'UKGOV' in actual.instrument_config.keys()
        assert 'USGOV' in actual.instrument_config.keys()
        assert 'EUROGOV' in actual.instrument_config.keys()
        assert actual.vol_target == VOL_TARGET

    def test_instantiation_with_input_vol_target(self, mock_instrument_config):
        """Test a Portfolio instantiates with an input vol_target."""
        input_vol = 50.5
        actual = Portfolio(mock_instrument_config, input_vol)
        assert actual.instrument_config == mock_instrument_config
        assert actual.vol_target == input_vol

    def test_instrument_properties(self, mock_instrument_config):
        """Test the properties derived from each Instrument are present."""
        actual = Portfolio(mock_instrument_config)
        expected_traded_instruments = ['FTSE100', 'FTSE250', 'FTSEAS']
        assert actual.traded_instruments == expected_traded_instruments
        assert list(actual.instruments.keys()) == expected_traded_instruments
        assert all([isinstance(inst, Instrument) for inst in actual.instruments.values()])
        assert list(actual.instrument_weights.keys()) == expected_traded_instruments
        assert list(actual.instrument_weights.values()) == [0.7 * 0.3 * 0.3333333333333333] * 3

    def test_untraded_instrument(self, mock_instrument_config):
        """Test an instrument with is_traded=False is not included in the Portfolio"""
        instrument_config = deepcopy(mock_instrument_config)
        instrument_config['FTSE250']['is_traded'] = False
        actual = Portfolio(instrument_config)
        expected_traded_instruments = ['FTSE100', 'FTSEAS']
        assert actual.traded_instruments == expected_traded_instruments
        assert list(actual.instruments.keys()) == expected_traded_instruments
        assert all([isinstance(inst, Instrument) for inst in actual.instruments.values()])

    def test_subsystems_list(self, mock_instrument_config):
        """Test the list of Subsystems in the Portfolio after running the run_subsystems method."""
        actual = Portfolio(mock_instrument_config)
        actual.run_subsystems()
        assert len(actual.subsystems) == 3
        assert all([isinstance(ss, Subsystem) for ss in actual.subsystems])
        assert all([not ss.subsystem_position.empty for ss in actual.subsystems])
        assert list(actual.subsystems[0].subsystem_position.columns) == ['FTSE100']
        assert list(actual.subsystems[1].subsystem_position.columns) == ['FTSE250']
        assert list(actual.subsystems[2].subsystem_position.columns) == ['FTSEAS']

    @pytest.mark.parametrize('rescale, expected_weight', [
        [False, 0.7 * 0.3 * 0.3333333333333333],
        [True, 0.3333333333333333]
    ])
    def test_pweights_list(self, mock_instrument_config, rescale, expected_weight):
        """Test the list of portfolio weights after running the run_subsystems method."""
        actual = Portfolio(mock_instrument_config)
        actual.run_subsystems(rescale=rescale)
        actual.pweights == [expected_weight] * 3

    def test_combine_subsystems(self, mock_instrument_config):
        """Test the combined positions after running the combine_subsystems method."""
        actual = Portfolio(mock_instrument_config)
        actual.run_subsystems()
        actual.combine_subsystems()
        expected_instruments = {'FTSE100', 'FTSE250', 'FTSEAS'}
        assert actual.diversification_multiplier >= 1
        assert actual.diversification_multiplier <= 2.5
        assert set(actual.target_position.columns) == expected_instruments
        assert not actual.target_position.dropna().empty


class TestSubsystem:

    def test_instantiation(self, mock_instrument):
        """Test a Subsystem instantiates with default vol_target."""
        actual = Subsystem(mock_instrument)
        assert actual.instrument_id == "FTSE100"
        assert actual.vol_target == VOL_TARGET

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
        assert actual.trading_capital == STARTING_CAPITAL
        assert not actual.converted_price.empty

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

    def test_returns(self, mock_instrument):
        """Test the returns of the Subsystem."""
        actual = Subsystem(mock_instrument)
        assert actual.subsystem_returns.shape[1] == 1
        set(actual.subsystem_returns.columns) == {'FTSE100'}
        assert not actual.subsystem_returns.dropna().empty


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

        expected_forecast_scalar = pd.Series({'price': 10 / 101})
        expected_scaled_df = pd.DataFrame(
            [7.920792079207921, 8.118811881188119, 8.316831683168317, 8.514851485148515,
             8.712871287128714, 8.910891089108912, 9.10891089108911, 9.306930693069308,
             9.504950495049506, 9.702970297029703, 9.900990099009901, 10.099009900990099,
             10.297029702970297, 10.495049504950495, 10.693069306930694, 10.891089108910892,
             11.08910891089109, 11.287128712871288, 11.485148514851486, 11.683168316831683,
             39.603960396039604, -29.702970297029704],
            index=pd.to_datetime(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                  '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                  '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                  '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                  '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28',
                                  '2019-10-29', '2019-01-30']),
            columns=['price']
        )
        expected_capped_df = pd.DataFrame(
            [7.920792079207921, 8.118811881188119, 8.316831683168317, 8.514851485148515,
             8.712871287128714, 8.910891089108912, 9.10891089108911, 9.306930693069308,
             9.504950495049506, 9.702970297029703, 9.900990099009901, 10.099009900990099,
             10.297029702970297, 10.495049504950495, 10.693069306930694, 10.891089108910892,
             11.08910891089109, 11.287128712871288, 11.485148514851486, 11.683168316831683,
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
            'open': 10 / 101,
            'high': 10 / 105,
            'low': 10 / 99,
            'close': 10 / 103,
            'volume': 10 / 10100
        })
        expected_scaled_df = pd.DataFrame({
            'open': [7.920792079207921, 8.118811881188119, 8.316831683168317, 8.514851485148515,
                     8.712871287128714, 8.910891089108912, 9.10891089108911, 9.306930693069308,
                     9.504950495049506, 9.702970297029703, 9.900990099009901, 10.099009900990099,
                     10.297029702970297, 10.495049504950495, 10.693069306930694,
                     10.891089108910892, 11.08910891089109, 11.287128712871288, 11.485148514851486,
                     11.683168316831683, 39.603960396039604, -29.702970297029704],
            'high': [8.0, 8.19047619047619, 8.38095238095238, 8.571428571428571, 8.761904761904761,
                     8.952380952380953, 9.142857142857142, 9.333333333333332, 9.523809523809524,
                     9.714285714285714, 9.904761904761905, 10.095238095238095, 10.285714285714285,
                     10.476190476190476, 10.666666666666666, 10.857142857142856,
                     11.047619047619047, 11.238095238095237, 11.428571428571427,
                     11.619047619047619, 38.476190476190474, -28.19047619047619],
            'low': [7.878787878787879, 8.080808080808081, 8.282828282828282, 8.484848484848484,
                    8.686868686868687, 8.88888888888889, 9.09090909090909, 9.292929292929292,
                    9.494949494949495, 9.696969696969697, 9.8989898989899, 10.1010101010101,
                    10.303030303030303, 10.505050505050505, 10.707070707070708, 10.909090909090908,
                    11.11111111111111, 11.313131313131313, 11.515151515151516, 11.717171717171716,
                    40.2020202020202, -30.505050505050505],
            'close': [7.961165048543689, 8.155339805825243, 8.349514563106796, 8.54368932038835,
                      8.737864077669903, 8.932038834951456, 9.12621359223301, 9.320388349514563,
                      9.514563106796116, 9.70873786407767, 9.902912621359222, 10.097087378640776,
                      10.291262135922329, 10.485436893203884, 10.679611650485437,
                      10.87378640776699, 11.067961165048544, 11.262135922330097, 11.45631067961165,
                      11.650485436893204, 39.029126213592235, -28.932038834951456],
            'volume': [7.920792079207921, 8.118811881188119, 8.316831683168317, 8.514851485148515,
                       8.712871287128714, 8.910891089108912, 9.10891089108911, 9.306930693069306,
                       9.504950495049505, 9.702970297029703, 9.900990099009901, 10.099009900990099,
                       10.297029702970297, 10.495049504950495, 10.693069306930694,
                       10.891089108910892, 11.08910891089109, 11.287128712871286,
                       11.485148514851485, 11.683168316831683, 39.603960396039604,
                       -29.702970297029704]
        },
            index=pd.to_datetime(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                                  '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
                                  '2019-01-11', '2019-01-14', '2019-01-15', '2019-01-16',
                                  '2019-01-17', '2019-01-18', '2019-01-21', '2019-01-22',
                                  '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-28',
                                  '2019-10-29', '2019-01-30'])
        )
        expected_capped_df = pd.DataFrame(
            [7.936307417149481, 8.13284956789715, 8.329391718644818, 8.525933869392485,
             8.722476020140155, 8.919018170887824, 9.115560321635492, 9.31210247238316,
             9.50864462313083, 9.705186773878497, 9.901728924626166, 10.098271075373834,
             10.294813226121503, 10.491355376869171, 10.68789752761684, 10.884439678364508,
             11.080981829112178, 11.277523979859845, 11.474066130607515, 11.670608281355182,
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

    def test_scalar_fxrate_and_trading_capital(self, mock_volatile_prices):
        """Test success when fx_rate and trading_capital inputs are scalars."""
        fx_rate = 1.
        vol_target = 20.
        trading_capital = 100000.
        actual = get_vol_scalar(mock_volatile_prices, fx_rate, vol_target, trading_capital)
        expected = pd.DataFrame(
            data=[np.nan, np.nan, 2.117698043838664, 2.6533901357227703, 3.368964180943644,
                  3.5622370845995386, 3.869430975824355, 4.379117212247568, 4.489797624822846,
                  4.7150459466450565, 3.218760906470086, 3.478338257294721, 3.7499263518950725,
                  3.5925446781315293, 3.8357531054424756, 4.089173148555686, 3.983260012294238,
                  4.096504773820595, 4.346444692731477, 4.611534885532055],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['vol_scalar']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_df_fxrate_and_trading_capital(self, mock_volatile_prices, mock_fx_rate, mock_trading_capital):
        """Test success when fx_rate and trading_capital inputs are DataFrames."""
        vol_target = 10.
        actual = get_vol_scalar(mock_volatile_prices, mock_fx_rate, vol_target, mock_trading_capital)
        expected = pd.DataFrame(
            data=[np.nan, np.nan, 0.1537858518827017, 0.1520664239779988, 0.19384143733852957,
                  0.1928354045286975, 0.29715328113331796, 0.3630182614412599, 0.432709871320629,
                  0.6070614068037926, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['vol_scalar']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_df_fxrate_scalar_trading_capital(self, mock_volatile_prices, mock_fx_rate):
        """Test success when fx_rate is a DataFrame and trading_capital is a scalar."""
        vol_target = 10.
        trading_capital = 20000
        actual = get_vol_scalar(mock_volatile_prices, mock_fx_rate, vol_target, trading_capital)
        expected = pd.DataFrame(
            data=[np.nan, np.nan, 0.2674536554481769, 0.33792538661777516, 0.43075874964117683,
                  0.45373036359693525, 0.49525546855552993, 0.5584896329865537, 0.576946495094172,
                  0.6070614068037926, 0.41430826444459856, 0.448239466146227, 0.486876960775782,
                  0.46176666814029943, 0.4944892491224025, 0.5298915574129436, 0.5210962862760646,
                  0.5375285098832955, 0.5740912287321988, 0.6068607560905455],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['vol_scalar']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_scalar_fxrate_df_trading_capital(self, mock_volatile_prices, mock_trading_capital):
        """Test success when trading_capital is a DataFrame."""
        vol_target = 10.
        fx_rate = 1.
        actual = get_vol_scalar(mock_volatile_prices, fx_rate, vol_target, mock_trading_capital)
        expected = pd.DataFrame(
            data=[np.nan, np.nan, 0.1217676375207232, 0.11940255610752466, 0.151603388142464,
                  0.1513950760954804, 0.2321658585494613, 0.28464261879609193, 0.3367348218617135,
                  0.4715045946645057, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan],
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
            columns=['vol_scalar']
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_empty_price_df_raises(self, mock_volatile_prices, mock_fx_rate, mock_trading_capital):
        """Test an error is raised when price is empty."""
        vol_target = 10.
        empty_price = mock_volatile_prices.head(0)
        with pytest.raises(InputDataError):
            get_vol_scalar(empty_price, mock_fx_rate, vol_target, mock_trading_capital)

    def test_empty_fxrate_df_raises(self, mock_volatile_prices, mock_fx_rate, mock_trading_capital):
        """Test an error is raised when fx_rate is empty."""
        vol_target = 10.
        empty_fx = mock_fx_rate.head(0)
        with pytest.raises(InputDataError):
            get_vol_scalar(mock_volatile_prices, empty_fx, vol_target, mock_trading_capital)

    def test_negative_vol_target_raises(self, mock_volatile_prices, mock_fx_rate, mock_trading_capital):
        vol_target = -69
        with pytest.raises(InputDataError):
            get_vol_scalar(mock_volatile_prices, mock_fx_rate, vol_target, mock_trading_capital)

    def test_empty_trading_capital_df_raises(self, mock_volatile_prices, mock_fx_rate, mock_trading_capital):
        """Test an error is raised when trading_capital is empty."""
        vol_target = 10.
        empty_trading_capital = mock_trading_capital.head(0)
        with pytest.raises(InputDataError):
            get_vol_scalar(mock_volatile_prices, mock_fx_rate, vol_target, empty_trading_capital)


class TestGetInstrumentValueVolatility:

    @pytest.mark.parametrize('fx_rate', [1., 0.5, 2.])
    def test_scalar_fx_rate(self, mock_volatile_prices, expected_instrument_value_vol, fx_rate):
        actual = get_instrument_value_volatility(mock_volatile_prices, fx_rate)
        expected = expected_instrument_value_vol * fx_rate
        pd.testing.assert_series_equal(actual, expected)

    def test_dataframe_fx_rate(self, mock_volatile_prices, mock_fx_rate, expected_instrument_value_vol):
        actual = get_instrument_value_volatility(mock_volatile_prices, mock_fx_rate)
        expected = expected_instrument_value_vol * mock_fx_rate.values.flatten()
        pd.testing.assert_series_equal(actual, expected)

    @pytest.mark.parametrize(
        'indices, expected_values', [
            [range(10),
             [np.nan, np.nan, 471.06537937805257, 372.8283303327153, 292.4796252535575,
              277.6718680913807, 254.39024032829258, 225.58728081668028, 218.37060930438275,
              207.53774866545393, 304.0145102588667, 281.32687169474906, 260.9517971269933,
              272.38353543033173, 255.1128797190944, np.nan, np.nan, np.nan, np.nan, np.nan]],
            [range(10, 20),
             [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
              304.09279389740385, 281.0733261685661, 258.7679595045838, 272.83943680288155,
              254.78442229702785, 237.76215323159047, 241.77519776641972, 234.3841402888492,
              219.45668452028062, 207.60636835601312]],
            [range(5, 17),
             [np.nan, np.nan, np.nan, np.nan, np.nan, 277.6718680913807, 254.39024032829258,
              225.58728081668028, 218.37060930438275, 207.53774866545393, 304.09279389740385,
              281.0733261685661, 258.7679595045838, 272.83943680288155, 254.78442229702785,
              237.76215323159047, 241.77519776641972, 235.0915061498443, 221.57269798877624,
              208.83577835417347]],
            [[4],
             [np.nan, np.nan, np.nan, np.nan, 292.4796252535575, 276.61083687972086,
              254.6507192637369, 225.0118764529086, 219.46498783983262, 208.98065305942004,
              np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
        ]
    )
    def test_dataframe_fx_rate_shorter_than_price_series(
            self, mock_volatile_prices, mock_fx_rate, indices, expected_values
    ):
        """Test when the fx_rate series is shorter than the price_df time series."""
        fx_rate = mock_fx_rate.iloc[indices]
        actual = get_instrument_value_volatility(mock_volatile_prices, fx_rate)
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
             [np.nan, np.nan, 471.06537937805257, 372.8283303327153, 292.4796252535575,
              277.6718680913807, 254.39024032829258, 225.58728081668028, 218.37060930438275,
              207.53774866545393]],
            [range(10, 20), pd.bdate_range('2019-01-15', periods=10),
             [np.nan, np.nan, 14.645536845328937, 184.11265518551676, 161.03573201742756,
              143.91012221282202, 162.09838119043027, 183.09346818990122, 164.87549110328996,
              152.2984835178301]],
            [range(5, 17), pd.bdate_range('2019-01-08', periods=12),
             [np.nan, np.nan, 115.58510528096002, 195.65630318716654, 178.52412009982365,
              331.0640444668015, 291.0185828720114, 258.87763573225664, 266.44343086739,
              245.66738065895228, 227.1536234194002, 228.2883080899905]],
            [[4], pd.bdate_range('2019-01-07', periods=1), [np.nan]]
        ]
    )
    def test_dataframe_fx_rate_longer_than_price_series(
            self, mock_volatile_prices, mock_fx_rate,
            indices, expected_timestamps, expected_values
    ):
        """Test when the fx_rate series is longer than the price_df time series."""
        price_subset = mock_volatile_prices.iloc[indices]
        actual = get_instrument_value_volatility(price_subset, mock_fx_rate)
        expected = pd.DataFrame(
            data=expected_values,
            index=pd.DatetimeIndex(expected_timestamps, name='timestamp'),
            columns=['instrument_value_volatility']
        )
        expected = expected.iloc[:, 0]
        pd.testing.assert_series_equal(actual, expected)

    def test_empty_price_raises(self, mock_volatile_prices, mock_fx_rate):
        empty_price = mock_volatile_prices.head(0)
        with pytest.raises(InputDataError):
            get_instrument_value_volatility(empty_price, mock_fx_rate)

    def test_empty_df_fxrate_raises(self, mock_volatile_prices, mock_fx_rate):
        empty_fx = mock_fx_rate.head(0)
        with pytest.raises(InputDataError):
            get_instrument_value_volatility(mock_volatile_prices, empty_fx)

    def test_unsupported_asset_class_raises(self, mock_volatile_prices):
        asset_class = "SNAKEOIL"
        expected_msg = "Block value calculations for SNAKEOIL are not yet implemented."
        with pytest.raises(NotImplementedError, match=expected_msg):
            get_instrument_value_volatility(mock_volatile_prices, 1., asset_class)


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
