from alphadog.framework.signals_config import (
    PARAMETERISED_STRATEGIES, get_price_df, DATA_FIXTURES
)


def test_parameterised_strategies():
    actual = PARAMETERISED_STRATEGIES
    expected_inner_keys = {'signal_func', 'raw_signal_func', 'required_data_fixtures',
                           'params', 'strategy_name', 'hierarchy_1'}
    assert isinstance(actual, dict)
    for inner_dict in actual.values():
        assert set(inner_dict.keys()).issuperset(expected_inner_keys)


def test_get_price_df():
    actual = get_price_df('FTSE100')
    assert all(actual.columns == ['close'])
    assert actual.index.dtype == 'datetime64[ns]'
    assert actual.index.name == 'timestamp'


def test_data_fixtures():
    """Test every strategy's required data fixtures are in the DATA_FIXTURES dict."""
    all_required_fixtures = []
    for strat_dict in PARAMETERISED_STRATEGIES.values():
        all_required_fixtures.extend(strat_dict['required_data_fixtures'])
    all_required_fixtures = set(all_required_fixtures)
    available_fixtures = set(DATA_FIXTURES)
    assert available_fixtures.issuperset(all_required_fixtures)
