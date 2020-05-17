import pytest

from alphadog.framework.config_handler import (
    BaseConfiguration, Strategy, Instrument,
    load_default_instrument_config, get_siblings, hierarchy_depth
)


class TestBaseConfiguration:

    def test_init(self, mock_instrument_dict):
        bc = BaseConfiguration(**mock_instrument_dict)
        assert bc.cost == 6
        assert bc.hierarchy_1 == "equity"
        assert bc.hierarchy_2 == "uk"

    def test_from_config(self, mock_instrument_dict):
        bc = BaseConfiguration.from_config(mock_instrument_dict)
        assert bc.cost == 6
        assert bc.hierarchy_1 == "equity"
        assert bc.hierarchy_2 == "uk"


class TestStrategy:

    def test_init(self, mock_strategy_dict):
        strat = Strategy(**mock_strategy_dict)
        assert strat.identifier == 'VMOM1'
        assert strat.hierarchy_1 == 'trend'
        assert strat.hierarchy_2 == 'VMOM'

    def test_from_identifier(self):
        strat = Strategy.from_identifier('VMOM1')
        assert strat.identifier == 'VMOM1'
        assert strat.hierarchy_1 == 'trend'
        assert strat.hierarchy_2 == 'VMOM'

    def test_from_config(self, mock_strategy_dict):
        strat = Strategy.from_config(mock_strategy_dict)
        assert strat.identifier == 'VMOM1'
        assert strat.hierarchy_1 == 'trend'
        assert strat.hierarchy_2 == 'VMOM'

    def test_reference_config(self):
        actual = Strategy.reference_config()
        assert isinstance(actual, dict)

    def test_weight_config(self, mock_strategy_dict):
        strat = Strategy.from_config(mock_strategy_dict)
        actual = strat.weight_config
        assert isinstance(actual, dict)

    def test_depth(self, mock_strategy_dict):
        strat = Strategy.from_config(mock_strategy_dict)
        actual = strat.depth()
        assert actual == 2

    def test_siblings(self, mock_strategy_dict):
        strat = Strategy.from_config(mock_strategy_dict)
        actual = strat.siblings()
        assert actual == [f"VMOM{k}" for k in range(1, 7)]


class TestInstrument:

    def test_init(self, mock_instrument_dict):
        instrument = Instrument(**mock_instrument_dict)
        assert instrument.identifier == 'FTSE100'
        assert instrument.cost == 6
        assert instrument.hierarchy_1 == "equity"
        assert instrument.hierarchy_2 == "uk"

    def test_from_identifier(self):
        instrument = Instrument.from_identifier('FTSE100')
        assert instrument.identifier == 'FTSE100'
        assert instrument.cost == 6
        assert instrument.hierarchy_1 == "equity"
        assert instrument.hierarchy_2 == "uk"

    def test_from_config(self, mock_instrument_dict):
        instrument = Instrument.from_config(mock_instrument_dict)
        assert instrument.identifier == 'FTSE100'
        assert instrument.cost == 6
        assert instrument.hierarchy_1 == "equity"
        assert instrument.hierarchy_2 == "uk"

    def test_reference_config(self):
        actual = Instrument.reference_config()
        assert isinstance(actual, dict)

    def test_weight_config(self, mock_instrument_dict):
        instrument = Instrument.from_config(mock_instrument_dict)
        actual = instrument.weight_config
        assert isinstance(actual, dict)

    def test_depth(self, mock_instrument_dict):
        instrument = Instrument.from_config(mock_instrument_dict)
        actual = instrument.depth()
        assert actual == 2

    def test_siblings(self, mock_instrument_dict):
        instrument = Instrument.from_config(mock_instrument_dict)
        actual = instrument.siblings()
        assert actual == ["FTSE100", "FTSE250", "FTSEAS"]

    def test_strategies(self, mock_instrument_dict):
        instrument = Instrument.from_config(mock_instrument_dict)
        actual = instrument.strategies
        assert isinstance(actual, dict)
        assert "BLONG" in actual.keys()
        assert isinstance(list(actual.values())[0], Strategy)

    def test_required_data_fixtures(self, mock_instrument_dict):
        instrument = Instrument.from_config(mock_instrument_dict)
        actual = instrument.required_data_fixtures
        assert isinstance(actual, list)
        assert 'price_df' in actual


class TestHierarchyDepth:

    def test_success(self, mock_instrument_dict):
        actual = hierarchy_depth(mock_instrument_dict)
        expected = 2
        assert actual == expected

    @pytest.mark.parametrize('name', ['test_instrument', None])
    def test_skipped_level_raises(self, mock_instrument_dict, name):
        instrument = mock_instrument_dict.copy()
        instrument['hierarchy_5'] = 'foo'
        expected_msg = f"{name} has skipped a level in its hierarchy"
        with pytest.raises(ValueError, match=expected_msg):
            hierarchy_depth(instrument, name)

    def test_no_hierarchy(self):
        no_hierarchy_dict = {'test': 'bar'}
        actual = hierarchy_depth(no_hierarchy_dict)
        expected = 0
        assert actual == expected


class TestGetSiblings:

    def test_object_at_level_1(self, mock_strategy_config):
        full_config = mock_strategy_config.copy()
        full_config['new_bias_strat'] = {'hierarchy_1': 'bias'}
        actual = get_siblings(full_config, 'BLONG')
        assert actual == ['BLONG', 'new_bias_strat']

    def test_object_at_level_2(self, mock_strategy_config):
        actual = get_siblings(mock_strategy_config, 'VMOM1')
        assert actual == ['VMOM1', 'VMOM2']

    @pytest.mark.parametrize('target', ['VBO1', 'BLONG'])
    def test_object_has_no_siblings(self, mock_strategy_config, target):
        actual = get_siblings(mock_strategy_config, target)
        assert actual == [target]


def test_load_default_instrument_config():
    actual = load_default_instrument_config()

    # Check some expected instruments are present
    expected_instruments = ['FTSE100', 'UKGOV', 'GLOBALCORP']
    assert set(actual.keys()).issuperset(expected_instruments)

    # Check a specific instrument has all the expected fields
    instrument = actual['UKGOV']
    expected_fields = [
        'traded_instrument', 'traded_info_link', 'cost', 'index', 'instrument_id',
        'yfinance_symbol', 'yfinance_link', 'currency', 'is_traded', 'hierarchy_1',
        'hierarchy_2', 'hierarchy_3', 'signals'
    ]
    assert set(instrument.keys()).issuperset(expected_fields)
