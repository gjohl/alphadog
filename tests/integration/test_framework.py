from alphadog.data.retrieval import PriceData
from alphadog.framework.config_handler import load_default_instrument_config, Instrument
from alphadog.framework.portfolio import Portfolio, Subsystem, Forecast
from alphadog.framework.signals_config import PARAMETERISED_STRATEGIES


def test_portfolio():
    """End-to-end test to check that the Portfolio runs and returns results."""
    instrument_config = load_default_instrument_config()
    pf = Portfolio(instrument_config, 20)
    pf.run_subsystems()
    pf.combine_subsystems()
    assert not pf.target_position.empty


def test_subsystem():
    """End-to-end test to check that the Subsystem runs and returns results."""
    inst = Instrument.from_identifier("SP500")
    ss = Subsystem(inst)
    ss.run_forecasts()
    ss.calc_position()
    assert not ss.subsystem_position.empty


def test_forecast():
    """End-to-end test to check that the Forecast runs and returns results."""
    signal_name = 'VMOM4'
    signal_func = PARAMETERISED_STRATEGIES[signal_name]['signal_func']
    prices = PriceData.from_instrument_id("FTSEEURO")
    fc = Forecast(signal_func, {'price_df': prices.df}, prices.name, f"{prices.name}|{signal_name}")
    assert not fc.scaled_forecast.empty
