import pytest


# TODO: move this to tests.mock
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
