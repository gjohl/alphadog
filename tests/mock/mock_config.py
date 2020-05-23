import pytest

from alphadog.framework.config_handler import Instrument


instrument_config = {
    "FTSE100": {
        "traded_instrument": "LEGAL & GENERAL UK 100 INDEX TRUST",
        "traded_info_link": "https://www.hl.co.uk/funds/fund-discounts,-prices--and--factsheets/search-results/l/legal-and-general-uk-100-index-class-c-accumulation",
        "cost": 6,
        "index": "FTSE 100",
        "instrument_id": "FTSE100",
        "yfinance_symbol": "^FTSE",
        "yfinance_link": "https://uk.finance.yahoo.com/quote/%5EFTSE?p=^FTSE",
        "currency": "GBP",
        "is_traded": True,
        "hierarchy_1": "equity",
        "hierarchy_2": "uk",
        "signals": ["VMOM1", "VMOM2", "VMOM3", "VMOM4", "VMOM5", "VMOM6", "VBO1", "VBO2", "VBO3", "VBO4", "VBO5", "VBO6", "BLONG"]
        },
    "FTSE250": {
        "traded_instrument": "HSBC FTSE 250 INDEX",
        "traded_info_link": "https://www.hl.co.uk/funds/fund-discounts,-prices--and--factsheets/search-results/h/hsbc-ftse-250-index-class-s-accumulation",
        "cost": 8,
        "index": "FTSE 250",
        "instrument_id": "FTSE250",
        "yfinance_traded_symbol": "0P0000WN7D.L",
        "yfinance_traded_link": "https://uk.finance.yahoo.com/quote/0P0000WN7D.L?p=0P0000WN7D.L",
        "yfinance_symbol": "^FTMC",
        "yfinance_link": "https://uk.finance.yahoo.com/quote/%5EFTMC?p=^FTMC",
        "currency": "GBP",
        "is_traded": True,
        "hierarchy_1": "equity",
        "hierarchy_2": "uk",
        "signals": ["VMOM1", "VMOM2", "VMOM3", "VMOM4", "VMOM5", "VMOM6", "VBO1", "VBO2", "VBO3", "VBO4", "VBO5", "VBO6", "BLONG"]
        },
    "FTSEAS": {
        "traded_instrument": "LEGAL & GENERAL UK INDEX",
        "traded_info_link": "https://www.hl.co.uk/funds/fund-discounts,-prices--and--factsheets/search-results/l/legal-and-general-uk-index-class-c-accumulation",
        "cost": 4,
        "index": "FTSE All Share",
        "instrument_id": "FTSEAS",
        "yfinance_symbol": "^FTAS",
        "yfinance_link": "https://uk.finance.yahoo.com/quote/%5EFTAS?p=^FTAS",
        "currency": "GBP",
        "is_traded": True,
        "hierarchy_1": "equity",
        "hierarchy_2": "uk",
        "signals": ["VMOM1", "VMOM2", "VMOM3", "VMOM4", "VMOM5", "VMOM6", "VBO1", "VBO2", "VBO3", "VBO4", "VBO5", "VBO6", "BLONG"]
    }
}


@pytest.fixture
def mock_instrument_config():
    return instrument_config


@pytest.fixture
def mock_instrument_dict():
    return instrument_config['FTSE100']


@pytest.fixture
def mock_instrument():
    return Instrument.from_config(instrument_config['FTSE100'])


def test_func1():
    return 1


def test_func2():
    return 2


strategy_config = {
    'VMOM1': {'signal_func': test_func1,
              'raw_signal_func': test_func1,
              'required_data_fixtures': ['price_df'],
              'params': {'fast': 2, 'slow': 8},
              'strategy_name': 'VMOM1',
              'hierarchy_1': 'trend',
              'hierarchy_2': 'VMOM'},
    'VMOM2': {'signal_func': test_func1,
              'raw_signal_func': test_func1,
              'required_data_fixtures': ['price_df'],
              'params': {'fast': 4, 'slow': 16},
              'strategy_name': 'VMOM2',
              'hierarchy_1': 'trend',
              'hierarchy_2': 'VMOM'},
    'VBO1': {'signal_func': test_func2,
             'raw_signal_func': test_func2,
             'required_data_fixtures': ['price_df'],
             'params': {'lookback': 10},
             'strategy_name': 'VBO1',
             'hierarchy_1': 'trend',
             'hierarchy_2': 'VBO'},
    'BLONG': {'signal_func': test_func1,
              'raw_signal_func': test_func1,
              'required_data_fixtures': ['price_df'],
              'params': {},
              'strategy_name': "BLONG",
              'hierarchy_1': 'bias'}
}

@pytest.fixture
def mock_strategy_config():
    return strategy_config


@pytest.fixture
def mock_strategy_dict():
    return strategy_config['VMOM1']


@pytest.fixture
def mock_instrument_weights_config():
    return {
        'equity': {
            'weight': 0.7,
            'uk': {'weight': 0.3},
            'europe': {'weight': 0.3},
            'us': {'weight': 0.4}
        },
        'bond': {
            'weight': 0.3,
        }
    }
