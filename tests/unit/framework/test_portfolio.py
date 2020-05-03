import pytest

from alphadog.framework.config_handler import hierarchy_depth


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
