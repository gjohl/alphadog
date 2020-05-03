from alphadog.framework.config_handler import load_default_instrument_config


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