"""
Strategies are parameterised signals ready to run on a data fixture.

The naming convention is to start the signal name with:
- "V": univariate signals on price data
- "X": cross-sectional signals
- "B": bias signals for long/short tilts
- "F": signals using fundamental data

This config defines the components of the strategies, in the following format.
{
    'VMOM1': {
        'signal_func': momentum_signal,
        'required_data_fixtures': ['price_df'],
        'params': {'fast': 2, 'slow': 8},
        'strategy_name': 'VMOM1'
    }
}

"""
from alphadog.signals.trend import momentum_signal, breakout_signal
from alphadog.signals.bias import long_bias_signal, short_bias_signal


PARAMETERISED_STRATEGIES = {}

# Momentum signals
for speed in range(1, 7):
    fast = 2 ** speed
    slow = 4 * fast
    sig = lambda price_df: momentum_signal(price_df, fast, slow)
    strategy_name = f"VMOM{speed}"

    temp_dict = {'signal_func': sig,
                 'raw_signal_func': momentum_signal,
                 'required_data_fixtures': ['price_df'],
                 'params': {'fast': fast, 'slow': slow},
                 'strategy_name': strategy_name,
                 'hierarchy_1': 'trend',
                 'hierarchy_2': 'VMOM'}
    PARAMETERISED_STRATEGIES[strategy_name] = temp_dict

# Breakout signals
for speed in range(1, 7):
    lookback = 10 * 2 ** (speed-1)
    sig = lambda price_df: breakout_signal(price_df, lookback)
    strategy_name = f"VBO{speed}"

    temp_dict = {'signal_func': sig,
                 'raw_signal_func': breakout_signal,
                 'required_data_fixtures': ['price_df'],
                 'params': {'lookback': lookback},
                 'strategy_name': strategy_name,
                 'hierarchy_1': 'trend',
                 'hierarchy_2': 'VBO'}
    PARAMETERISED_STRATEGIES[strategy_name] = temp_dict

# Bias signals
PARAMETERISED_STRATEGIES["BLONG"] = {
    'signal_func': long_bias_signal,
    'raw_signal_func': long_bias_signal,
    'required_data_fixtures': ['price_df'],
    'params': {},
    'strategy_name': "BLONG",
    'hierarchy_1': 'bias',
}

PARAMETERISED_STRATEGIES["BSHORT"] = {
    'signal_func': short_bias_signal,
    'raw_signal_func': short_bias_signal,
    'required_data_fixtures': ['price_df'],
    'params': {},
    'strategy_name': "BSHORT",
    'hierarchy_1': 'bias',
}
