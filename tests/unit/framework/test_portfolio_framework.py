import pytest

from alphadog.framework.portfolio import (
    load_default_instrument_config, hierarchy_depth, get_siblings
)


@pytest.fixture()
def mock_instrument():
    return {
        "instrument_id": "test_instrument",
        "hierarchy_1": "equity",
        "hierarchy_2": "uk",
        "signals": ["VMOM1", "VMOM2", "VMOM3", "VMOM4", "VMOM5", "VMOM6",
                    "VBO1", "VBO2", "VBO3", "VBO4", "VBO5","VBO6",
                    "BLONG"]
    }


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
