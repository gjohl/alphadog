import pytest

from alphadog.framework.config_handler import (
    BaseConfiguration, Strategy, Instrument,
    load_default_instrument_config, get_siblings, hierarchy_depth
)


def test_base_configuration():
    # TODO TEST
    pass


def test_strategy():
    # TODO TEST
    pass


def test_instrument():
    # TODO TEST
    pass


class TestHierarchyDepth:

    def test_success(self, mock_instrument):
        actual = hierarchy_depth(mock_instrument)
        expected = 2
        assert actual == expected

    def test_skipped_level_raises(self, mock_instrument):
        instrument = mock_instrument.copy()
        instrument['hierarchy_5'] = 'foo'
        expected_msg = "Instrument test_instrument has skipped a level in its hierarchy"
        with pytest.raises(ValueError, match=expected_msg):
            hierarchy_depth(instrument)

    def test_no_hierarchy(self, mock_instrument):
        no_hierarchy_dict = {'test': 'bar'}
        actual = hierarchy_depth(no_hierarchy_dict)
        expected = 0
        assert actual == expected


class TestGetSiblings:
    # TODO TEST

    def object_at_level_1(self):
        pass

    def object_at_level_2(self):
        pass

    def object_has_no_siblings(self):
        pass


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
